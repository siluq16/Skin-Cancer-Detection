import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import gc # <--- MỚI: Thư viện dọn rác bộ nhớ

# --- CẤU HÌNH ---
st.set_page_config(page_title="Chẩn đoán Ung thư da", page_icon="🩺")
st.title("🩺 Hệ thống Chẩn đoán Ung thư Da")

# Ép chạy CPU để tránh lỗi CUDA trên Cloud và tiết kiệm VRAM ảo
device = torch.device('cpu') 

@st.cache_resource # <--- QUAN TRỌNG: Giữ model trong cache để không load lại
def load_ensemble_models():
    def load_single_model(arch_type, path):
        model = None
        try:
            # map_location=device là CPU
            checkpoint = torch.load(path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            if arch_type == 'resnet50':
                model = models.resnet50(weights=None)
                model.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(model.fc.in_features, 7))
            elif arch_type == 'densenet121':
                model = models.densenet121(weights=None)
                if 'classifier.weight' in state_dict:
                     model.classifier = nn.Linear(model.classifier.in_features, 7)
                else:
                     model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.classifier.in_features, 7))
            elif arch_type == 'efficientnet_b4':
                model = models.efficientnet_b4(weights=None)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
            
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            return None

    # Load 3 model
    m_res = load_single_model('resnet50', 'skin_resnet50.pth')
    m_dense = load_single_model('densenet121', 'best_densenet121.pth')
    m_eff = load_single_model('efficientnet_b4', 'best_efficientnet_b4.pth')
    
    return m_res, m_dense, m_eff

# Load models
with st.spinner('Đang khởi động AI...'):
    model_resnet, model_dense, model_eff = load_ensemble_models()

if model_resnet and model_dense and model_eff:
    st.success("✅ Hệ thống sẵn sàng!")

# --- XỬ LÝ ẢNH ---
labels_map = { 0: 'AKIEC (Dày sừng quang hóa)', 1: 'BCC (Ung thư biểu mô tế bào đáy)', 2: 'BKL (Tổn thương lành tính)', 3: 'DF (U xơ da)', 4: 'MEL (Ung thư hắc tố - Nguy hiểm)', 5: 'NV (Nốt ruồi lành tính)', 6: 'VASC (Tổn thương mạch máu)' }

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    # SỬA LỖI WARNING: Dùng use_container_width thay vì use_column_width
    st.image(image, caption='Ảnh tải lên', use_container_width=True) 
    
    if st.button('🔍 Phân tích'):
        with st.spinner('Đang xử lý...'):
            # Transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)), # Dùng ảnh nhỏ 224 cho tất cả để tiết kiệm RAM (chấp nhận giảm nhẹ độ chính xác EfficientNet)
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Inference (Dùng with torch.no_grad để không tốn RAM lưu gradient)
            with torch.no_grad():
                out_res = model_resnet(img_tensor)
                prob_res = F.softmax(out_res, dim=1)
                
                out_dense = model_dense(img_tensor)
                prob_dense = F.softmax(out_dense, dim=1)

                # Resize lên 380 cho EfficientNet (nếu RAM chịu nổi)
                # Hoặc dùng luôn ảnh 224 cho EfficientNet để tránh crash (chấp nhận hy sinh chút độ chính xác)
                img_380 = F.interpolate(img_tensor, size=(380, 380), mode='bilinear')
                out_eff = model_eff(img_380)
                prob_eff = F.softmax(out_eff, dim=1)

                final_prob = (prob_res * 0.4) + (prob_dense * 0.3) + (prob_eff * 0.3)
                top_p, top_class = torch.max(final_prob, 1)
                
                # --- DỌN RÁC NGAY LẬP TỨC ---
                del img_tensor, img_380, out_res, out_dense, out_eff
                gc.collect()

            pred_idx = top_class.item()
            confidence = top_p.item() * 100

        st.info(f"Kết quả: **{labels_map[pred_idx]}** ({confidence:.2f}%)")
        st.bar_chart(final_prob.detach().numpy()[0])