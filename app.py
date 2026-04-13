import streamlit as st
import cv2
import numpy as np
from rembg import remove, new_session
import gc

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="AI 智能服装换色器", layout="wide", page_icon="🎨")
st.title("🎨 AI 智能服装换色器 (极速防崩溃版)")
st.write("已针对云端服务器进行内存优化，极速提取，告别崩溃。")

# ==========================================
# 模型缓存机制 (使用 4MB 的超轻量级模型 u2netp)
# ==========================================
@st.cache_resource
def get_rembg_session():
    """使用 u2netp (仅4MB)，大幅降低内存占用，防止云端部署崩溃"""
    return new_session("u2netp")

# ==========================================
# 核心渲染与图像处理函数
# ==========================================
def render_exact_color(orig_img, mask_3d, target_lab):
    """确保零色差，保留褶皱"""
    orig_lab = cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_t, a_t, b_t = target_lab

    l_orig = orig_lab[:, :, 0]
    mask_bool = mask_3d[:, :, 0] > 0.5
    
    if not np.any(mask_bool):
        return orig_img

    l_mean_orig = np.mean(l_orig[mask_bool])

    l_new = l_orig - l_mean_orig + l_t
    l_new = np.clip(l_new, 0, 255).astype(np.uint8)
    a_new = np.full_like(l_orig, a_t, dtype=np.uint8)
    b_new = np.full_like(l_orig, b_t, dtype=np.uint8)

    new_lab = cv2.merge([l_new, a_new, b_new])
    new_bgr = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

    final_out = new_bgr.astype(np.float32) * mask_3d + orig_img.astype(np.float32) * (1.0 - mask_3d)
    return np.clip(final_out, 0, 255).astype(np.uint8)

def get_lab_metrics(img_bgr):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    h, w = img_bgr.shape[:2]
    return np.mean(img_lab[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)], axis=(0, 1))

def generate_ai_mask(orig_bgr, orig_bytes, shape):
    """提取蒙版，并加入防内存溢出机制"""
    session = get_rembg_session() 
    
    # 1. 提取人体蒙版
    output_data = remove(orig_bytes, session=session)
    nparr = np.frombuffer(output_data, np.uint8)
    rgba_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    person_mask = rgba_img[:, :, 3]
    person_mask = cv2.resize(person_mask, (shape[1], shape[0]))
    _, person_mask = cv2.threshold(person_mask, 200, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)

    # 2. 特征过滤 (降低膨胀迭代次数以节省计算量)
    lower_skin = np.array([0, 10, 30], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_mask = cv2.dilate(skin_mask, np.ones((5, 5), np.uint8), iterations=2)

    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    dark_mask = cv2.dilate(dark_mask, np.ones((5, 5), np.uint8), iterations=1)

    white_mask = cv2.inRange(hsv, np.array([0, 0, 210]), np.array([180, 30, 255]))
    white_mask = cv2.dilate(white_mask, np.ones((5, 5), np.uint8), iterations=1)

    exclude_mask = cv2.bitwise_or(skin_mask, dark_mask)
    exclude_mask = cv2.bitwise_or(exclude_mask, white_mask)

    clothes_mask = cv2.bitwise_and(person_mask, cv2.bitwise_not(exclude_mask))

    clothes_mask = cv2.morphologyEx(clothes_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    clothes_mask = cv2.morphologyEx(clothes_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(clothes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(clothes_mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        clothes_mask = clean_mask

    clothes_mask = cv2.erode(clothes_mask, np.ones((3, 3), np.uint8), iterations=1)
    clothes_mask = cv2.GaussianBlur(clothes_mask, (5, 5), 0)

    mask_3d = np.repeat((clothes_mask.astype(np.float32) / 255.0)[:, :, np.newaxis], 3, axis=2)
    
    # 强制清理本函数的临时大矩阵
    del rgba_img, hsv, skin_mask, dark_mask, white_mask, exclude_mask
    gc.collect() 
    
    return mask_3d

# ==========================================
# UI 交互层
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 上传原图")
    orig_file = st.file_uploader("选择原图", type=['jpg', 'jpeg', 'png'], key="orig")

with col2:
    st.subheader("2. 上传参考图")
    ref_files = st.file_uploader("选择颜色参考图", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="refs")

if orig_file and ref_files:
    orig_bytes = orig_file.getvalue()
    orig_bgr = cv2.imdecode(np.frombuffer(orig_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # 图像预处理：如果原图过大，强制缩小以防 OOM
    max_dim = 1200
    h, w = orig_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        orig_bgr = cv2.resize(orig_bgr, (int(w * scale), int(h * scale)))
        is_success, buffer = cv2.imencode(".jpg", orig_bgr)
        orig_bytes = buffer.tobytes()

    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    
    st.markdown("---")
    st.subheader("预览原图")
    st.image(orig_rgb, width=400)

    if st.button("🚀 开始极速换色", type="primary"):
        with st.spinner('正在处理中 (已启用防崩溃机制)...'):
            try:
                shape = orig_bgr.shape[:2]
                mask_3d = generate_ai_mask(orig_bgr, orig_bytes, shape)
                st.success("✅ 蒙版提取成功！")
                st.markdown("### 🎨 生成结果")
                
                for ref_file in ref_files:
                    ref_bytes = ref_file.getvalue()
                    ref_bgr = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                    target_lab = get_lab_metrics(ref_bgr)
                    final_img = render_exact_color(orig_bgr, mask_3d, target_lab)
                    
                    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

                    res_col1, res_col2 = st.columns([1, 3])
                    with res_col1:
                        st.image(ref_rgb, caption=f"参考: {ref_file.name}", use_container_width=True)
                    with res_col2:
                        st.image(final_rgb, caption="结果", use_container_width=True)
                        is_success, buffer = cv2.imencode(".jpg", final_img)
                        st.download_button(
                            label=f"⬇️ 下载",
                            data=buffer.tobytes(),
                            file_name=f"result_{ref_file.name}",
                            mime="image/jpeg"
                        )
                    st.divider()
                    
                    # 循环内及时释放内存
                    del ref_bgr, final_img, final_rgb, ref_rgb
                    gc.collect()
                    
            except Exception as e:
                st.error(f"处理发生错误: {str(e)}")
