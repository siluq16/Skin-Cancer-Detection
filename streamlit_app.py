import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import gc
import os

# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="Chẩn đoán Ung thư da", page_icon="🩺")
st.title("🩺 Hệ thống Chẩn đoán Ung thư Da")
st.caption("🚀 Phiên bản High-Res (380px) - Chế độ tiết kiệm RAM")

device = torch.device('cpu')

# --- 2. HÀM LOAD & PREDICT TUẦN TỰ (QUAN TRỌNG) ---
# Không dùng @st.cache_resource cho model nữa vì ta cần xóa nó ngay sau khi dùng
def predict_one_model(arch_type, path, img_tensor):
    model = None
    prob = None
    try:
        # Load Model
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        if arch_type == 'resnet50':
            model = models.resnet50(weights=None)
            model.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(model.fc.in_features, 7))
        elif arch_type == 'densenet121':
            model = models.densenet121(weights=None)
            if 'classifier.weight' in state_dict:
                 model.classifier = nn.Linear(model.classifier.in_features, 7)
            else:
                 model.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(model.classifier.in_features, 7))
        elif arch_type == 'efficientnet_b4':
            model = models.efficientnet_b4(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        # Dự đoán
        with torch.no_grad():
            out = model(img_tensor)
            prob = F.softmax(out, dim=1)
            
    except Exception as e:
        st.error(f"Lỗi khi chạy {arch_type}: {e}")
        return None
    finally:
        # --- DỌN RÁC CỰC MẠNH ---
        del model
        del checkpoint
        if 'state_dict' in locals(): del state_dict
        torch.cuda.empty_cache() # Dù chạy CPU vẫn gọi cho chắc
        gc.collect() # Ép dọn RAM ngay lập tức
        
    return prob

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


uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ảnh đã tải lên', use_container_width=True)

    if st.button('🔍 Phân tích chi tiết (380px)'):
        progress_bar = st.progress(0, text="Đang khởi tạo...")
        
        try:
            final_prob = torch.zeros(1, 7).to(device)
            models_ran = 0
            
            # --- GIAI ĐOẠN 1: RESNET50 ---
            progress_bar.progress(10, text="Đang chạy ResNet50 (1/3)...")
            # ResNet thường train ở 224, dùng 380 cũng được nhưng tốn RAM, ta resize về 224 cho nó nhẹ bớt
            # để dành RAM cho EfficientNet sau cùng.
            t_res = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_res = t_res(image).unsqueeze(0).to(device)
            
            prob_res = predict_one_model('resnet50', 'skin_resnet50.pth', img_res)
            if prob_res is not None:
                final_prob += prob_res * 0.4
                models_ran += 1
            del img_res, prob_res
            gc.collect()

            # --- GIAI ĐOẠN 2: DENSENET121 ---
            progress_bar.progress(40, text="Đang chạy DenseNet121 (2/3)...")
            # DenseNet cũng chạy 224
            img_dense = t_res(image).unsqueeze(0).to(device) # Tái sử dụng transform 224
            
            prob_dense = predict_one_model('densenet121', 'best_densenet121.pth', img_dense)
            if prob_dense is not None:
                final_prob += prob_dense * 0.3
                models_ran += 1
            del img_dense, prob_dense
            gc.collect()

            # --- GIAI ĐOẠN 3: EFFICIENTNET-B4 (BOSS CUỐI - 380PX) ---
            progress_bar.progress(70, text="Đang chạy EfficientNet-B4 (3/3) - High Res...")
            # Đây là lúc dùng 380px như bạn muốn
            t_eff = transforms.Compose([
                transforms.Resize((400, 400)), # Resize to hơn chút
                transforms.CenterCrop(380),    # Crop đúng chuẩn 380
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_eff = t_eff(image).unsqueeze(0).to(device)
            
            prob_eff = predict_one_model('efficientnet_b4', 'best_efficientnet_b4.pth', img_eff)
            if prob_eff is not None:
                final_prob += prob_eff * 0.3
                models_ran += 1
            del img_eff, prob_eff
            gc.collect()

            progress_bar.progress(100, text="Hoàn tất!")

            if models_ran == 0:
                st.error("❌ Lỗi: Không chạy được model nào!")
                st.stop()

            # Lấy kết quả
            top_p, top_class = torch.max(final_prob, 1)
            pred_idx = top_class.item()
            confidence = top_p.item() * 100

            # Hiển thị
            st.divider()
            if pred_idx in [1, 4]:
                st.error(f"### ⚠ KẾT QUẢ: {labels_map[pred_idx]}")
            elif pred_idx in [2, 5]:
                st.success(f"### 🎉 KẾT QUẢ: {labels_map[pred_idx]}")
            else:
                st.warning(f"### ℹ KẾT QUẢ: {labels_map[pred_idx]}")
                
            st.info(f"Độ tin cậy: **{confidence:.2f}%**")
            st.bar_chart(final_prob.detach().numpy()[0])
            
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")