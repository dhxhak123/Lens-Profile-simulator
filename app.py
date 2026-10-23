import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

# 페이지 기본 설정
st.set_page_config(page_title="Lithography & Profile Tools", layout="wide")

# ==========================================
# 1. 사이드바 네비게이션
# ==========================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Tool", [
    "2D Wedge Prism Inspector", 
    "3D Lens Simulator", 
    "3D Grayscale Simulator"
])
st.sidebar.markdown("---")

# ==========================================
# 2. 첫 번째 툴: 2D 웻지 프리즘 검사기
# ==========================================
if app_mode == "2D Wedge Prism Inspector":
    st.title("WLO Wedge Prism Profile Comparison Tool")
    st.write("음각(Negative) 비대칭 웻지 프리즘 측정 데이터를 디자인 스펙과 비교합니다.")

    st.sidebar.header("Design Specifications")
    slope_width = st.sidebar.number_input("Slope Width (um)", value=50.0)
    steep_width = st.sidebar.number_input("Steep Width (um)", value=10.0)
    depth = st.sidebar.number_input("Depth / Sag (um)", value=50.0)

    total_w = slope_width + steep_width
    start_x = -total_w / 2
    x_coords = [start_x, start_x + slope_width, start_x + total_w]
    y_coords = [0, -depth, 0]
    ideal_x = [start_x - 20] + x_coords + [start_x + total_w + 20]
    ideal_y = [0] + y_coords + [0]

    uploaded_file_prism = st.file_uploader("측정 데이터 CSV 업로드 (첫 번째 열 X, 두 번째 열 Y)", type=['csv'])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ideal_x, ideal_y, color='red', linestyle='--', linewidth=2, label='Design Spec')

    if uploaded_file_prism is not None:
        try:
            df = pd.read_csv(uploaded_file_prism)
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            st.sidebar.markdown("---")
            st.sidebar.header("Alignment Options")
            align_x = st.sidebar.checkbox("X축 중앙 정렬 (Center X)", value=True)
            align_y = st.sidebar.checkbox("Y축 표면 정렬 (Align Top to 0)", value=True)
            
            x_data = df[x_col].copy()
            y_data = df[y_col].copy()
            
            if align_x:
                x_data = x_data - (x_data.max() + x_data.min()) / 2
            if align_y:
                y_data = y_data - y_data.max()
                
            ax.plot(x_data, y_data, color='blue', linewidth=1.5, label='Measured Profile')
        except Exception as e:
            st.error(f"CSV 로드 오류: {e}")

    ax.set_xlabel("X Position (um)")
    ax.set_ylabel("Depth / Sag (um)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.axis('equal') 
    
    st.pyplot(fig, use_container_width=True)

# ==========================================
# 3. 두 번째 툴: 신규 2D 렌즈 검사기
# ==========================================
elif app_mode == "3D Lens Simulator":
    st.title("Micro-Lens Profile Comparison Tool")
    st.write("마이크로 렌즈 측정 데이터(CSV)를 직경(Dia)과 깊이(Sag) 기반 디자인 스펙과 비교합니다.")

    st.sidebar.header("Lens Specifications")
    lens_dia = st.sidebar.number_input("Diameter (Dia, ㎛)", value=4.0, step=0.1)
    lens_sag = st.sidebar.number_input("Sag (Height/Depth, ㎛)", value=2.0, step=0.1)
    lens_tone = st.sidebar.radio("Lens Tone", ["Concave (음각, 아래로 파임)", "Convex (양각, 위로 볼록)"])

    # 곡률 반경(R) 계산
    r = lens_dia / 2
    if lens_sag > 0:
        R_val = (r**2 + lens_sag**2) / (2 * lens_sag)
    else:
        R_val = 0.0

    st.sidebar.markdown("---")
    st.sidebar.success(f"💡 Calculated RoC (R값): **{R_val:.3f} ㎛**")

    # 이상적인 렌즈 곡선(Circle Eq) 생성
    x_lens = np.linspace(-r, r, 500)
    if "Concave" in lens_tone:
        y_lens = -np.sqrt(R_val**2 - x_lens**2) + R_val - lens_sag
    else:
        y_lens = np.sqrt(R_val**2 - x_lens**2) - R_val + lens_sag

    # 좌우 평탄면(여백) 추가
    margin = lens_dia * 0.5
    x_left = np.linspace(-r - margin, -r, 50)
    y_left = np.zeros_like(x_left)
    x_right = np.linspace(r, r + margin, 50)
    y_right = np.zeros_like(x_right)

    ideal_x = np.concatenate([x_left, x_lens, x_right])
    ideal_y = np.concatenate([y_left, y_lens, y_right])

    uploaded_file_lens = st.file_uploader("측정 데이터 CSV 업로드 (첫 번째 열 X, 두 번째 열 Y)", type=['csv'])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ideal_x, ideal_y, color='red', linestyle='--', linewidth=2, 
            label=f'Design Spec\n(Dia: {lens_dia}㎛, Sag: {lens_sag}㎛, R: {R_val:.2f}㎛)')

    if uploaded_file_lens is not None:
        try:
            df = pd.read_csv(uploaded_file_lens)
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            st.sidebar.markdown("---")
            st.sidebar.header("Alignment Options")
            align_x = st.sidebar.checkbox("X축 중앙 정렬 (Center X)", value=True)
            align_y = st.sidebar.checkbox("Y축 표면 정렬 (Align Top to 0)", value=True)
            
            x_data = df[x_col].copy()
            y_data = df[y_col].copy()
            
            if align_x:
                x_data = x_data - (x_data.max() + x_data.min()) / 2
            if align_y:
                y_data = y_data - y_data.max()
                
            ax.plot(x_data, y_data, color='blue', linewidth=1.5, label='Measured Profile')
        except Exception as e:
            st.error(f"CSV 로드 오류: {e}")

    ax.set_xlabel("X Position (㎛)")
    ax.set_ylabel("Depth / Height (㎛)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.axis('equal') 
    
    st.pyplot(fig, use_container_width=True)

# ==========================================
# 4. 세 번째 툴: 신규 10-bit 3D 시뮬레이터
# ==========================================
elif app_mode == "3D Grayscale Simulator":
    st.title("10-bit Grayscale Lithography 3D Simulator")

    st.sidebar.header("3D Process Parameters")
    
    pixel_preset = st.sidebar.radio("Pixel Size (nm/pixel)", ("100 nm", "500 nm", "직접 입력"))
    if pixel_preset == "100 nm":
        pixel_size_nm = 100.0
    elif pixel_preset == "500 nm":
        pixel_size_nm = 500.0
    else:
        pixel_size_nm = st.sidebar.number_input("해상도 직접 입력 (nm)", value=100.0, step=10.0)

    target_height = st.sidebar.number_input("Target Lens Height (㎛)", value=10.0, step=0.1)
    pr_tone = st.sidebar.radio("Photoresist Tone", ("Positive", "Negative"))
    
    st.sidebar.markdown("---")
    st.sidebar.header("Image View / Crop")
    enable_crop = st.sidebar.checkbox("이미지 일부만 잘라서 확대 보기 (Crop)", value=True)
    if enable_crop:
        crop_size = st.sidebar.number_input("가로/세로 픽셀 크기", value=100, step=50)
        start_x = st.sidebar.number_input("시작 X 좌표", value=0, step=50)
        start_y = st.sidebar.number_input("시작 Y 좌표", value=0, step=50)
    
    apply_blur = st.sidebar.checkbox("Simulate Laser Spot / Blur", value=False)
    if apply_blur:
        sigma_value = st.sidebar.slider("Spot Size (Sigma)", 0.1, 5.0, 1.0)

    uploaded_file_3d = st.file_uploader("Upload 10-bit Grayscale PNG Design", type=["png", "jpg"])

    if uploaded_file_3d is not None:
        image = Image.open(uploaded_file_3d)
        img_array = np.array(image, dtype=np.float32)
        MAX_GRAY = 1023.0
        img_array = np.clip(img_array, 0, MAX_GRAY)

        step = 1  
        
        if enable_crop:
            end_x = min(start_x + int(crop_size), img_array.shape[1])
            end_y = min(start_y + int(crop_size), img_array.shape[0])
            img_array = img_array[int(start_y):end_y, int(start_x):end_x]
        else:
            max_size = 500
            if img_array.shape[0] > max_size or img_array.shape[1] > max_size:
                step = max(img_array.shape[0] // max_size, img_array.shape[1] // max_size)
                img_array = img_array[::step, ::step]

        if pr_tone == "Positive":
            z_data = ((MAX_GRAY - img_array) / MAX_GRAY) * target_height
        else:
            z_data = (img_array / MAX_GRAY) * target_height

        if apply_blur:
            z_data = gaussian_filter(z_data, sigma=sigma_value)

        pixel_size_um = (pixel_size_nm * step) / 1000.0
        x_um = np.arange(img_array.shape[1]) * pixel_size_um
        y_um = np.arange(img_array.shape[0]) * pixel_size_um

        st.subheader("3D Simulated Profile (True Scale)")
        
        fig_3d = go.Figure(data=[go.Surface(
            z=z_data, x=x_um, y=y_um,
            colorscale='Viridis', cmin=0, cmax=target_height
        )])
        
        fig_3d.update_layout(
            scene=dict(
                zaxis=dict(range=[0, target_height], title='Z / Height (㎛)'),
                xaxis=dict(title='X (㎛)'),
                yaxis=dict(title='Y (㎛)'),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30), height=700
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)