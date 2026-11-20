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
    "2D Lens Inspector", 
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
    
    # 공통으로 사용하는 Depth 먼저 입력받기
    depth = st.sidebar.number_input("Depth / Sag (um)", value=50.0, step=1.0)
    
    # 입력 방식 선택 메뉴 추가
    input_method = st.sidebar.radio("스펙 입력 방식", ["Width (폭) 직접 입력", "Angle (각도) 계산"])
    
    if input_method == "Width (폭) 직접 입력":
        slope_width = st.sidebar.number_input("Slope Width (um)", value=50.0)
        steep_width = st.sidebar.number_input("Steep Width (um)", value=10.0)
    else:
        # 각도 입력 시 자동 계산 로직
        slope_angle = st.sidebar.number_input("완만한 경사각 (Slope Angle, °)", value=45.0, step=1.0)
        steep_angle = st.sidebar.number_input("가파른 경사각 (Steep Angle, °)", value=78.0, step=1.0)
        
        # 삼각함수 Width = Depth / tan(θ) 적용
        slope_width = depth / np.tan(np.radians(slope_angle)) if slope_angle < 90 else 0.0
        steep_width = depth / np.tan(np.radians(steep_angle)) if steep_angle < 90 else 0.0
        
        # 자동 계산된 값을 사이드바에 안내
        st.sidebar.info(f"💡 자동 계산된 폭:\n- Slope Width: {slope_width:.2f} ㎛\n- Steep Width: {steep_width:.2f} ㎛")

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
            # UTF-8과 CP949 자동 감지 (한글 깨짐 방지)
            try:
                df = pd.read_csv(uploaded_file_prism, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file_prism.seek(0)
                df = pd.read_csv(uploaded_file_prism, encoding='cp949')

            x_col = df.columns[0]
            y_col = df.columns[1]
            
            st.sidebar.markdown("---")
            st.sidebar.header("Alignment Options")
            align_x = st.sidebar.checkbox("X축 중앙 정렬 (Center X)", value=True)
            align_y = st.sidebar.checkbox("Y축 표면 정렬 (Align Top to 0)", value=True)
            
            # 수동 X축 이동 기능 유지
            x_shift = st.sidebar.number_input("수동 X축 이동 (X Shift, ㎛)", value=0.0, step=1.0)
            
            x_data = df[x_col].copy()
            y_data = df[y_col].copy()
            
            if align_x:
                x_data = x_data - (x_data.max() + x_data.min()) / 2
            if align_y:
                y_data = y_data - y_data.max()
                
            # 수동 입력한 값만큼 X축 이동 적용
            x_data = x_data + x_shift
                
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
# 3. 두 번째 툴: 신규 2D 렌즈 검사기 (비구면 Conic K 지원)
# ==========================================
elif app_mode == "2D Lens Inspector":
    st.title("Micro-Lens Profile Comparison Tool")
    st.write("마이크로 렌즈 측정 데이터(CSV)를 직경(Dia), 깊이(Sag)/곡률반경(RoC), Conic 상수(K) 기반 디자인 스펙과 비교합니다.")

    # --- 헬퍼 함수 ---
    def load_csv_safe(file_obj):
        encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]
        last_err = None
        for enc in encodings:
            try:
                file_obj.seek(0)
                return pd.read_csv(file_obj, encoding=enc)
            except UnicodeDecodeError as e:
                last_err = e
        raise last_err

    def calc_conic_sag(r_arr, roc_val, k_val):
        disc = 1.0 - (1.0 + k_val) * (r_arr / roc_val) ** 2
        if np.any(disc < -1e-9):
            return None
        disc = np.maximum(disc, 0.0)
        return (r_arr ** 2 / roc_val) / (1.0 + np.sqrt(disc))

    # --- 1. 사이드바: Lens Specifications ---
    st.sidebar.header("Lens Specifications")

    spec_mode = st.sidebar.radio(
        "스펙 입력 방식",
        ["Dia + Sag + K (R 자동 계산)", "Dia + RoC + K (Sag 자동 계산)"]
    )

    lens_dia = st.sidebar.number_input("Diameter (Dia, μm)", min_value=0.1, value=100.0, step=1.0, format="%.2f")
    r_max = lens_dia / 2.0

    conic_k = st.sidebar.number_input(
        "Conic Constant (K)",
        value=0.0000,
        step=0.1000,
        format="%.4f",
        help="0: 구면, -1: 포물면, -1 미만: 쌍곡면, -1~0: 장축타원, 0 초과: 단축타원"
    )

    valid_spec = True
    error_msg = ""

    if spec_mode == "Dia + Sag + K (R 자동 계산)":
        lens_sag = st.sidebar.number_input("Sag (Height/Depth, μm)", min_value=0.01, value=40.0, step=1.0, format="%.2f")
        k_min = -1.0 - (r_max / lens_sag) ** 2
        k_max = (r_max / lens_sag) ** 2 - 1.0

        if conic_k <= k_min or conic_k > k_max:
            valid_spec = False
            error_msg = (
                f"현재 Dia({lens_dia:.1f} μm), Sag({lens_sag:.1f} μm) 조건에서 입력 가능한 "
                f"Conic K 범위는 ({k_min:.3f} 초과 ~ {k_max:.3f} 이하)입니다."
            )
            R_val = np.nan
        else:
            R_val = (r_max ** 2 + (1.0 + conic_k) * (lens_sag ** 2)) / (2.0 * lens_sag)
    else:
        R_val = st.sidebar.number_input("Radius of Curvature (RoC, μm)", min_value=0.01, value=51.25, step=1.0, format="%.3f")
        edge_sag = calc_conic_sag(np.array([r_max]), R_val, conic_k)
        if edge_sag is None:
            valid_spec = False
            error_msg = "입력한 Dia, RoC, Conic K 조합에서는 렌즈 외곽이 정의되지 않습니다."
            lens_sag = np.nan
        else:
            lens_sag = float(edge_sag[0])

    lens_tone = st.sidebar.radio("Lens Tone", ["Concave (음각, 아래로 파임)", "Convex (양각, 위로 볼록)"])

    if valid_spec:
        if spec_mode.startswith("Dia + Sag"):
            st.sidebar.success(f"💡 Calculated RoC (R값): **{R_val:.3f} μm**")
        else:
            st.sidebar.success(f"💡 Calculated Sag: **{lens_sag:.3f} μm**")
    else:
        st.sidebar.error(error_msg)

    # --- 2. CSV 업로드 ---
    uploaded_file = st.file_uploader("측정 데이터 CSV 업로드 (첫 번째 열 X, 두 번째 열 Y)", type=["csv"])

    x_meas, y_meas = None, None
    if uploaded_file is not None:
        try:
            df = load_csv_safe(uploaded_file)
            raw_x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            raw_y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            valid_rows = raw_x.notna() & raw_y.notna()
            x_meas = raw_x[valid_rows].to_numpy()
            y_meas = raw_y[valid_rows].to_numpy()

            order = np.argsort(x_meas)
            x_meas, y_meas = x_meas[order], y_meas[order]
        except Exception as e:
            st.error(f"CSV 로드 오류: {e}")

    # --- 3. 측정 데이터 정렬 옵션 ---
    if x_meas is not None and len(x_meas) > 0:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Alignment Settings")
        auto_center_x = st.sidebar.checkbox("X축 중앙 자동 정렬", value=True)
        auto_zero_y = st.sidebar.checkbox("Y축 기준면(0) 자동 정렬", value=True)
        x_offset = st.sidebar.number_input("X 미세 오프셋 (μm)", value=0.0, step=0.5, format="%.2f")
        y_offset = st.sidebar.number_input("Y 미세 오프셋 (μm)", value=0.0, step=0.5, format="%.2f")

        if auto_center_x:
            if "Concave" in lens_tone:
                center_x = x_meas[np.argmin(y_meas)]
            else:
                center_x = x_meas[np.argmax(y_meas)]
            x_meas = x_meas - center_x

        if auto_zero_y:
            edge_n = max(1, int(len(y_meas) * 0.1))
            baseline = np.median(np.concatenate([y_meas[:edge_n], y_meas[-edge_n:]]))
            y_meas = y_meas - baseline

        x_meas = x_meas + x_offset
        y_meas = y_meas + y_offset

    # --- 4. 디자인 프로파일 생성 및 그래프 출력 ---
    if not valid_spec:
        st.warning("사이드바의 렌즈 스펙(Dia, Sag/RoC, Conic K)을 유효한 범위로 조정해 주세요.")
    else:
        margin = r_max * 0.5
        x_design = np.linspace(-(r_max + margin), r_max + margin, 1000)
        y_design = np.zeros_like(x_design)

        in_lens = np.abs(x_design) <= r_max
        r_vals = np.abs(x_design[in_lens])
        z_profile = calc_conic_sag(r_vals, R_val, conic_k)

        if "Concave" in lens_tone:
            y_design[in_lens] = z_profile - lens_sag
        else:
            y_design[in_lens] = lens_sag - z_profile

        fig, ax = plt.subplots(figsize=(10, 5))

        spec_label = (
            rf"Design Spec (Dia: {lens_dia:.1f} $\mu\mathrm{{m}}$, "
            rf"Sag: {lens_sag:.1f} $\mu\mathrm{{m}}$, "
            rf"R: {R_val:.2f} $\mu\mathrm{{m}}$, K: {conic_k:.3f})"
        )
        ax.plot(x_design, y_design, "r--", linewidth=2, label=spec_label)

        if x_meas is not None and len(x_meas) > 0:
            ax.plot(x_meas, y_meas, "b-", linewidth=1.5, alpha=0.85, label="Measured Profile (CSV)")

        ax.set_xlabel(r"Position ($\mu\mathrm{m}$)")
        ax.set_ylabel(r"Depth / Height ($\mu\mathrm{m}$)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")
        ax.set_aspect("equal", adjustable="datalim")

        st.pyplot(fig)
        plt.close(fig)

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