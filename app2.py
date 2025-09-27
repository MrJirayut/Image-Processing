import streamlit as st
import numpy as np
import cv2
from skimage import color, img_as_float, restoration
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fundus Image Enhancement", layout="wide")

# ---------------- Utility Functions ----------------
def show_histogram(image, is_gray=False):
    fig, ax = plt.subplots()
    if is_gray or image.ndim == 2:
        ax.hist(image.ravel(), bins=256, range=(0,1), color="black")
        ax.set_title("Histogram (Gray)")
    else:
        colors = ("r","g","b")
        for i, col in enumerate(colors):
            ax.hist(image[..., i].ravel(), bins=256, range=(0,1), color=col, alpha=0.5)
        ax.set_title("Histogram (RGB)")
    return fig

def tv_decomposition(image, weight):
    """Split into base + detail using TV denoising."""
    base = restoration.denoise_tv_chambolle(image, weight=weight, channel_axis=-1)
    detail = image - base
    return base, detail

def naka_rushton_adaptation(base, n=1.0):
    hsv = color.rgb2hsv(base)
    L = hsv[...,2]
    Mg, Sg = np.mean(L), np.std(L)
    sigma_g = Mg / (1 + np.exp(Sg))
    L_out = (L**n) / (L**n + sigma_g**n)
    hsv[...,2] = np.clip(L_out, 0, 1)
    return color.hsv2rgb(hsv)

def fuse_layers(base_corr, detail, alpha_r, alpha_g, alpha_b):
    fused = np.zeros_like(base_corr)
    # Gaussian smoothing for weighting
    kernel = cv2.getGaussianKernel(21, 10)
    kernel = kernel @ kernel.T
    for c, alpha in enumerate([alpha_r, alpha_g, alpha_b]):
        if alpha == 0: 
            fused[...,c] = base_corr[...,c]
        else:
            weight = cv2.filter2D(np.abs(detail[...,c]), -1, kernel)
            fused[...,c] = base_corr[...,c] + alpha * weight * detail[...,c]
    return np.clip(fused, 0, 1)

# ---------------- Streamlit UI ----------------
st.title("👁 Retinal Fundus Image Enhancement (based on Wang et al. 2021)")

uploaded = st.file_uploader("Upload fundus image", type=["png","jpg","jpeg"])
if uploaded:
    # Load and normalize
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = img_as_float(rgb)

    st.sidebar.header("Parameters")
    tv_weight = st.sidebar.slider("TV decomposition weight (λ₂)", 0.05, 1.0, 0.3, 0.05)
    alpha = st.sidebar.slider("Detail enhancement (α for R,G)", 0.0, 1000.0, 600.0, 50.0)
    alpha_b = st.sidebar.slider("Detail weight for B channel", 0.0, 200.0, 0.0, 10.0)

    # --- Processing pipeline ---
    base, detail = tv_decomposition(img, tv_weight)
    base_corr = naka_rushton_adaptation(base)
    enhanced = fuse_layers(base_corr, detail, alpha, alpha, alpha_b)

    # --- Display ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(rgb, channels="RGB")
        st.pyplot(show_histogram(img))

    with col2:
        st.subheader("Enhanced Image")
        st.image((enhanced*255).astype(np.uint8), channels="RGB")
        st.pyplot(show_histogram(enhanced))
else:
    st.info("Please upload a fundus image to start.")
