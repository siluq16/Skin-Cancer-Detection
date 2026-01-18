import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Chẩn đoán Ung thư da", page_icon="🩺")
st.title("🩺 Hệ thống Chẩn đoán Ung thư Da (Ensemble AI)")
st.write("Tải lên hình ảnh vết thương da để hệ thống phân tích.")

# --- 2. LOAD MODEL (Sử dụng Cache để không phải load lại mỗi lần) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_ensemble_models():
    # Khai báo hàm load model (Code đã sửa chuẩn của bạn)
    def load_single_model(arch_type, path):
        model = None
        try:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
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
            st.error(f"Lỗi load {arch_type}: {e}")
            return None

    # ĐỔI ĐƯỜNG DẪN TỚI FILE TRÊN MÁY BẠN
    m_res = load_single_model('resnet50', 'skin_resnet50.pth')
    m_dense = load_single_model('densenet121', 'best_densenet121.pth') # File DenseNet mới train lại
    m_eff = load_single_model('efficientnet_b4', 'best_efficientnet_b4.pth')
    
    return m_res, m_dense, m_eff

# Load model ngay khi vào app
with st.spinner('Đang khởi động "Tam giác vàng" AI... Vui lòng đợi!'):
    model_resnet, model_dense, model_eff = load_ensemble_models()

if model_resnet and model_dense and model_eff:
    st.success("✅ Hệ thống đã sẵn sàng!")

# --- 3. XỬ LÝ ẢNH ---
labels_map = {
    0: 'AKIEC (Dày sừng quang hóa)',
    1: 'BCC (Ung thư biểu mô tế bào đáy)',
    2: 'BKL (Tổn thương lành tính)',
    3: 'DF (U xơ da)',
    4: 'MEL (Ung thư hắc tố - Nguy hiểm)',
    5: 'NV (Nốt ruồi lành tính)',
    6: 'VASC (Tổn thương mạch máu)'
}

def process_image(image):
    # Transform cho EfficientNet (380)
    transform_eff = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform_eff(image).unsqueeze(0).to(device)
    return img_tensor

# --- 4. GIAO DIỆN CHÍNH ---
uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    
    if st.button('🔍 Phân tích ngay'):
        with st.spinner('Đang hội chẩn 3 chuyên gia AI...'):
            img_tensor = process_image(image)
            
            # 1. EfficientNet (Chạy ảnh gốc 380)
            out_eff = model_eff(img_tensor)
            prob_eff = F.softmax(out_eff, dim=1)

            # 2. ResNet & DenseNet (Resize xuống 224)
            img_small = F.interpolate(img_tensor, size=(224, 224), mode='bilinear')
            
            out_res = model_resnet(img_small)
            prob_res = F.softmax(out_res, dim=1)
            
            out_dense = model_dense(img_small)
            prob_dense = F.softmax(out_dense, dim=1)

            # 3. Ensemble (Weighted Average)
            # Trọng số bạn có thể tùy chỉnh
            final_prob = (prob_res * 0.4) + (prob_dense * 0.3) + (prob_eff * 0.3)
            
            # Lấy kết quả
            top_p, top_class = torch.max(final_prob, 1)
            pred_idx = top_class.item()
            confidence = top_p.item() * 100

        # Hiển thị kết quả
        st.markdown("---")
        if pred_idx in [1, 4]: # Các lớp Ung thư nguy hiểm
            st.error(f"### ⚠ KẾT QUẢ: {labels_map[pred_idx]}")
        elif pred_idx in [2, 5]: # Lành tính
            st.success(f"### 🎉 KẾT QUẢ: {labels_map[pred_idx]}")
        else:
            st.warning(f"### ℹ KẾT QUẢ: {labels_map[pred_idx]}")
            
        st.info(f"Độ tin cậy: **{confidence:.2f}%**")
        
        # Show chi tiết xác suất
        st.write("Chi tiết xác suất các lớp:")
        probs = final_prob.detach().cpu().numpy()[0]
        st.bar_chart({labels_map[i]: probs[i] for i in range(7)})