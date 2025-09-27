import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, img_as_ubyte

st.set_page_config(page_title="Image Processing Demo", layout="wide")

# Utility Functions ---------------------------------------------------------

def plot_histogram(image, is_gray=True):
    """Return a matplotlib figure showing histogram of an image."""
    fig, ax = plt.subplots()
    if is_gray or len(image.shape) == 2:
        ax.hist(image.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7)
        ax.set_title("Histogram (Grayscale)")
    else:
        colors = ("b", "g", "r")
        for i, col in enumerate(colors):
            ax.hist(image[:, :, i].ravel(), bins=256, range=(0, 256),
                    color=col, alpha=0.5, label=f"Channel {col.upper()}")
        ax.set_title("Histogram (Color)")
        ax.legend()
    ax.set_xlim([0, 256])
    return fig


def apply_processing(method, image, params):
    """Apply selected image processing method with given params."""
    gray = len(image.shape) == 2

    if method == "Linear Negative":
        return 255 - image

    elif method == "Contrast Stretching":
        r1, s1, r2, s2 = params
        def contrast_stretch(pixel):
            if pixel < r1:
                return (s1 / r1) * pixel
            elif pixel < r2:
                return ((s2 - s1) / (r2 - r1)) * (pixel - r1) + s1
            else:
                return ((255 - s2) / (255 - r2)) * (pixel - r2) + s2
        lut = np.array([np.clip(contrast_stretch(i), 0, 255) for i in range(256)]).astype("uint8")
        return cv2.LUT(image, lut)

    elif method == "Piecewise Linear Transformation":
        points = params  # list of (r, s)
        xp, fp = zip(*points)
        lut = np.interp(np.arange(256), xp, fp).astype("uint8")
        return cv2.LUT(image, lut)

    elif method == "Log Transformation":
        c = params
        img_float = image.astype(float)
        log_img = c * np.log1p(img_float)
        log_img = 255 * log_img / np.max(log_img)
        return log_img.astype("uint8")

    elif method == "Gamma Transformation":
        gamma = params
        invGamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, lut)

    elif method == "Histogram Equalization":
        if gray:
            return cv2.equalizeHist(image)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    elif method == "Adaptive Histogram Equalization":
        clip_limit, grid_size = params
        if gray:
            return exposure.equalize_adapthist(image, clip_limit=clip_limit, nbins=256).astype("float32")
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = img_as_ubyte(exposure.equalize_adapthist(l, clip_limit=clip_limit))
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    elif method == "CLAHE":
        clip_limit, tile_grid = params
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        if gray:
            return clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


# Streamlit UI --------------------------------------------------------------

st.title("📷 Image Processing Interactive Demo")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

img_type = st.sidebar.radio("Image type", ["Grayscale", "Color"])

method = st.sidebar.selectbox(
    "Select Processing Method",
    [
        "Linear Negative",
        "Contrast Stretching",
        "Piecewise Linear Transformation",
        "Log Transformation",
        "Gamma Transformation",
        "Histogram Equalization",
        "Adaptive Histogram Equalization",
        "CLAHE",
    ]
)

# Load Image
if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_type == "Grayscale":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Parameters per method
    params = None
    if method == "Contrast Stretching":
        r1 = st.sidebar.slider("r1", 0, 255, 50)
        s1 = st.sidebar.slider("s1", 0, 255, 0)
        r2 = st.sidebar.slider("r2", 0, 255, 200)
        s2 = st.sidebar.slider("s2", 0, 255, 255)
        params = (r1, s1, r2, s2)

    elif method == "Piecewise Linear Transformation":
        st.sidebar.write("Define control points (r,s).")
        p1 = (st.sidebar.slider("r1", 0, 255, 0), st.sidebar.slider("s1", 0, 255, 0))
        p2 = (st.sidebar.slider("r2", 0, 255, 128), st.sidebar.slider("s2", 0, 255, 128))
        p3 = (st.sidebar.slider("r3", 0, 255, 255), st.sidebar.slider("s3", 0, 255, 255))
        params = [p1, p2, p3]

    elif method == "Log Transformation":
        c = st.sidebar.slider("c", 1, 50, 30)
        params = c

    elif method == "Gamma Transformation":
        gamma = st.sidebar.slider("Gamma", 0.1, 5.0, 1.0, 0.1)
        params = gamma

    elif method == "Adaptive Histogram Equalization":
        clip_limit = st.sidebar.slider("Clip Limit", 0.01, 0.1, 0.03, 0.01)
        grid_size = st.sidebar.slider("Grid Size", 4, 16, 8)
        params = (clip_limit, grid_size)

    elif method == "CLAHE":
        clip_limit = st.sidebar.slider("Clip Limit", 1.0, 10.0, 2.0)
        tile_grid = st.sidebar.slider("Tile Grid Size", 2, 16, 8)
        params = (clip_limit, tile_grid)

    # Process image
    processed = apply_processing(method, image, params)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, channels="BGR" if img_type == "Color" else "GRAY")
        st.pyplot(plot_histogram(image, is_gray=(img_type == "Grayscale")))

    with col2:
        st.subheader(f"Processed Image ({method})")
        st.image(processed, channels="BGR" if img_type == "Color" else "GRAY")
        st.pyplot(plot_histogram(processed, is_gray=(img_type == "Grayscale")))
else:
    st.info("Please upload an image to start.")
