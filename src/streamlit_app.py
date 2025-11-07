import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T
import time

# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="AI Object Detection | Faster R-CNN",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# DARK ELEGANT THEME
# -------------------------
st.markdown("""
<style>
/* Background gradient */
.stApp {
    background: radial-gradient(circle at top left, #0f0f1a, #0a0814, #0a0610);
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13112a, #0e0b22);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
    color: #f5f5f5;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #e6e6ff;
    font-weight: 600;
    text-shadow: 0 0 12px rgba(180, 160, 255, 0.25);
}
.subtitle {
    color: #b3b3c6;
    text-align: center;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Section dividers */
hr {
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Cards */
.card {
    background: rgba(26, 24, 40, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 0 15px rgba(136, 97, 255, 0.12);
    transition: all 0.3s ease;
}
.card:hover {
    box-shadow: 0 0 25px rgba(166, 128, 255, 0.25);
    transform: translateY(-2px);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #8b5cf6, #7c3aed, #a855f7);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.3rem;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.4);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #7c3aed, #9333ea);
    box-shadow: 0 0 35px rgba(147, 51, 234, 0.6);
    transform: scale(1.04);
}

/* Upload area */
[data-testid="stFileUploaderDropzone"] {
    background-color: rgba(28, 24, 45, 0.8);
    border: 1px dashed rgba(200, 180, 255, 0.3);
    border-radius: 12px;
    color: #e0e0ff;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: rgba(26, 24, 40, 0.8) !important;
    color: white !important;
    border-radius: 6px;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    background: rgba(18, 16, 28, 0.6);
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #b794f4;
}
[data-testid="stMetricLabel"] {
    color: #d1c9ff;
}

/* Sidebar metrics */
.css-1v3fvcr, .css-1l269bu {
    color: #d1c9ff !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# MODEL LOADING
# -------------------------
@st.cache_resource
def load_model():
    with st.spinner("🔮 Initializing Faster R-CNN Model..."):
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        categories = weights.meta["categories"]
        preprocess = T.Compose([T.ToTensor()])
        return model, categories, preprocess, device

# -------------------------
# DETECTION FUNCTION
# -------------------------
def detect_objects(image, conf_thresh=0.6, max_size=640):
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    tensor = preprocess(image).to(device)
    with torch.no_grad():
        outputs = model([tensor])[0]

    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    frame = np.array(image)
    # Elegant glowing bounding boxes
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        glow_color = (180 + (i * 5) % 50, 100 + (i * 20) % 80, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), glow_color, 2)
        text = f"{CATEGORIES[label]}: {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 255), 2)
    return frame

# -------------------------
# LOAD MODEL
# -------------------------
model, CATEGORIES, preprocess, device = load_model()

# -------------------------
# HEADER
# -------------------------
st.markdown("""
<h1 style='text-align: center;'>💜 AI Object Detection System</h1>
<p class='subtitle'>Powered by Faster R-CNN | Dark Elegant Edition</p>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("⚙️ Control Panel")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.95, 0.6, 0.05)
mode = st.sidebar.radio("Input Mode", ["📷 Live Camera", "📁 Upload Image"])
st.sidebar.markdown("---")

# -------------------------
# CAMERA MODE
# -------------------------
if mode == "📷 Live Camera":
    st.sidebar.subheader("🎥 Camera Settings")
    cam_index = st.sidebar.selectbox("Select Camera", [0, 1, 2], index=0)
    start = st.sidebar.button("▶️ Start Detection", use_container_width=True)
    stop = st.sidebar.button("⏹️ Stop", use_container_width=True)

    st.markdown("## 🎬 Live Detection Feed")
    stframe = st.empty()

    if start:
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            st.error("❌ Could not open camera.")
        else:
            st.success("Camera activated. Detecting in real-time...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame = detect_objects(frame, conf_thresh=conf_threshold)
                stframe.image(frame, channels="RGB", use_column_width=True)
                if stop:
                    break
            cap.release()
            st.info("🛑 Detection stopped.")
    else:
        st.info("Press ▶️ to start live detection.")

# -------------------------
# IMAGE UPLOAD MODE
# -------------------------
else:
    st.markdown("## 🖼️ Image Upload Detection")
    file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("🔍 Run Detection", type="primary", use_container_width=True):
            with st.spinner("✨ Detecting objects..."):
                result = detect_objects(image, conf_thresh=conf_threshold)
                st.image(result, caption="Detected Objects", use_container_width=True)
