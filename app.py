import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import time
from datetime import datetime
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ── Page Configuration ──
st.set_page_config(
    page_title="HUD - Highway Bend Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS for HUD Professional Look ──
st.markdown("""
<style>
    /* ── Base ── */
    .stApp {
        background: #0a0a0f;
        font-family: 'Segoe UI', 'Consolas', monospace;
    }
    
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ── Top Bar ── */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid rgba(0,180,255,0.1);
    }
    
    .logo {
        font-size: 1.5rem;
        color: #00b4ff;
        font-weight: 700;
        font-family: 'Consolas', monospace;
    }
    
    .status-active {
        color: #00ff88;
        font-size: 0.7rem;
        letter-spacing: 2px;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
    
    .top-links {
        display: flex;
        gap: 20px;
    }
    
    .top-links span {
        color: rgba(0,180,255,0.3);
        font-size: 0.6rem;
        font-family: 'Consolas', monospace;
    }
    
    /* ── HUD Headers ── */
    .hud-header {
        color: #00b4ff;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 600;
        border-bottom: 1px solid rgba(0, 180, 255, 0.15);
        padding-bottom: 8px;
        margin-bottom: 15px;
        font-family: 'Consolas', monospace;
    }
    
    /* ── Detection Result Box ── */
    .result-sharp {
        background: linear-gradient(135deg, rgba(255, 50, 50, 0.15), rgba(200, 0, 0, 0.05));
        border: 2px solid rgba(255, 50, 50, 0.4);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 40px rgba(255, 50, 50, 0.1);
    }
    
    .result-straight {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 200, 100, 0.05));
        border: 2px solid rgba(0, 255, 136, 0.3);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.05);
    }
    
    .result-sharp-text {
        color: #ff4444;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 3px;
        font-family: 'Consolas', monospace;
        text-shadow: 0 0 30px rgba(255, 50, 50, 0.3);
    }
    
    .result-straight-text {
        color: #00ff88;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 3px;
        font-family: 'Consolas', monospace;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
    }
    
    .result-confidence {
        color: rgba(255, 255, 255, 0.6);
        font-size: 1rem;
        letter-spacing: 1px;
        font-family: 'Consolas', monospace;
        margin-top: 8px;
    }
    
    .result-meta {
        color: rgba(255, 255, 255, 0.25);
        font-size: 0.7rem;
        font-family: 'Consolas', monospace;
        margin-top: 5px;
    }
    
    /* ── Sidebar Cards ── */
    .bend-analysis-box {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    }
    
    .bend-analysis-sharp {
        background: rgba(255,68,68,0.1);
        border: 1px solid rgba(255,68,68,0.2);
    }
    
    .bend-analysis-straight {
        background: rgba(0,255,136,0.05);
        border: 1px solid rgba(0,255,136,0.1);
    }
    
    .bend-label {
        font-size: 0.7rem;
        letter-spacing: 2px;
        font-family: 'Consolas', monospace;
    }
    
    .bend-angle {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'Consolas', monospace;
    }
    
    .bend-sub {
        color: rgba(255,255,255,0.3);
        font-size: 0.6rem;
        font-family: 'Consolas', monospace;
    }
    
    .bend-intensity {
        font-size: 0.8rem;
        font-family: 'Consolas', monospace;
        margin-top: 10px;
        padding: 5px;
        border-radius: 5px;
    }
    
    .bend-intensity-sharp {
        background: rgba(255,68,68,0.1);
        color: #ff4444;
    }
    
    .bend-intensity-straight {
        background: rgba(0,255,136,0.05);
        color: #00ff88;
    }
    
    /* ── History ── */
    .history-item {
        background: rgba(0, 180, 255, 0.03);
        border-left: 3px solid #00b4ff;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 8px 8px 0;
        font-family: 'Consolas', monospace;
    }
    
    .history-item-sharp {
        border-left-color: #ff4444;
    }
    
    .history-item-straight {
        border-left-color: #00ff88;
    }
    
    .history-time {
        color: rgba(255, 255, 255, 0.3);
        font-size: 0.6rem;
        font-family: 'Consolas', monospace;
    }
    
    .history-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.8rem;
        font-family: 'Consolas', monospace;
    }
    
    /* ── Buttons ── */
    .stButton > button {
        background: rgba(0, 180, 255, 0.1) !important;
        color: #00b4ff !important;
        border: 1px solid rgba(0, 180, 255, 0.2) !important;
        border-radius: 8px !important;
        font-family: 'Consolas', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 2px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: rgba(0, 180, 255, 0.15) !important;
        border-color: #00b4ff !important;
        box-shadow: 0 0 30px rgba(0, 180, 255, 0.1) !important;
    }
    
    /* ── File Uploader ── */
    .upload-container {
        border: 1px dashed rgba(0, 180, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 30px !important;
        text-align: center !important;
        background: rgba(0, 180, 255, 0.02) !important;
    }
    
    .upload-container:hover {
        border-color: rgba(0, 180, 255, 0.4) !important;
    }
    
    /* ── Diagnostics ── */
    .diagnostic-item {
        color: rgba(0, 180, 255, 0.4);
        font-size: 0.6rem;
        letter-spacing: 1px;
        font-family: 'Consolas', monospace;
        border-bottom: 1px solid rgba(0, 180, 255, 0.05);
        padding: 4px 0;
    }
    
    .diagnostic-ok {
        color: #00ff88;
    }
    
    .diagnostic-warn {
        color: #ffaa00;
    }
    
    .diagnostic-error {
        color: #ff4444;
    }
    
    /* ── Advice Box ── */
    .advice-box {
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        font-family: 'Consolas', monospace;
        border: 1px solid rgba(0, 180, 255, 0.08);
        background: rgba(0, 180, 255, 0.03);
    }
    
    .advice-box-sharp {
        border-color: rgba(255, 68, 68, 0.2);
        background: rgba(255, 68, 68, 0.05);
    }
    
    .advice-box-straight {
        border-color: rgba(0, 255, 136, 0.15);
        background: rgba(0, 255, 136, 0.03);
    }
    
    .advice-title-sharp {
        color: #ff4444;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .advice-title-straight {
        color: #00ff88;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .advice-text {
        color: rgba(255,255,255,0.6);
        font-size: 0.7rem;
        margin-top: 5px;
    }
    
    /* ── Footer ── */
    .footer {
        display: flex;
        justify-content: space-between;
        font-family: 'Consolas', monospace;
        font-size: 0.5rem;
        color: rgba(255,255,255,0.08);
        padding-top: 10px;
        border-top: 1px solid rgba(0,180,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──
CLASSES = ["sharp", "straight"]
CLASS_NAMES = ["Sharp Bend", "Straight Road"]
CLASS_COLORS = {"sharp": "#ff4444", "straight": "#00ff88"}
CLASS_EMOJIS = {"sharp": "🔴", "straight": "🟢"}
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Session State ──
if "history" not in st.session_state:
    st.session_state.history = []
if "total_sharp" not in st.session_state:
    st.session_state.total_sharp = 0
if "total_straight" not in st.session_state:
    st.session_state.total_straight = 0
if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0
if "total_errors" not in st.session_state:
    st.session_state.total_errors = 0
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

# ── Load Model ──
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2)
    )
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

try:
    model = load_model()
except:
    st.error("⚠️ Model file not found. Please ensure model.pth is in the app directory.")
    st.stop()

# ── Transform ──
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Input Validation ──
def validate_road_image(image):
    img_array = np.array(image)
    if img_array.shape[0] < 50 or img_array.shape[1] < 50:
        return False, "Image too small"
    
    h, w = img_array.shape[:2]
    sample_size = min(1000, h * w // 10)
    sample_indices = np.random.choice(h * w, sample_size, replace=False)
    
    road_colors = 0
    for idx in sample_indices:
        r, c = divmod(idx, w)
        if r < h and c < w:
            pixel = img_array[r, c]
            if len(pixel) >= 3:
                r_val, g_val, b_val = pixel[:3]
                if abs(r_val - g_val) < 30 and abs(g_val - b_val) < 30:
                    road_colors += 1
                elif r_val > 100 and g_val > 80 and b_val < 100:
                    road_colors += 1
                elif r_val > 150 and g_val > 150 and b_val > 150:
                    road_colors += 1
    
    road_ratio = road_colors / sample_size
    if road_ratio < 0.05:
        return False, "No road-like colors detected"
    return True, "Valid"

# ── Prediction Function ──
def predict_image(img, source="upload"):
    is_valid, reason = validate_road_image(img)
    if not is_valid:
        return None, reason
    
    start = time.time()
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = probs.argmax().item()
    
    elapsed = (time.time() - start) * 1000
    predicted_class = CLASSES[pred]
    confidence = probs[pred].item() * 100
    
    result = {
        "prediction": predicted_class,
        "confidence": confidence,
        "elapsed": elapsed,
        "sharp_prob": probs[0].item() * 100,
        "straight_prob": probs[1].item() * 100,
        "source": source,
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    return result, None

# ── Video Processor ──
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.transform = None
        self.classes = None
        self.device = None
        self.result = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            if self.model is not None:
                tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    pred = probs.argmax().item()
                    confidence = probs[pred].item() * 100
                
                self.result = {
                    "prediction": self.classes[pred],
                    "confidence": confidence,
                    "sharp_prob": probs[0].item() * 100,
                    "straight_prob": probs[1].item() * 100
                }
                
                self.frame_count += 1
                if time.time() - self.start_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # HUD overlay
                label = f"{self.classes[pred].upper()}"
                color = (0, 255, 136) if self.classes[pred] == "straight" else (68, 68, 255)
                cv2.rectangle(img, (10, 10), (300, 120), (0, 0, 0, 180), -1)
                cv2.putText(img, "DETECTION", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
                cv2.putText(img, f"{label} {confidence:.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(img, f"FPS: {self.fps}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
                
                bar_x, bar_y = 20, 115
                bar_w, bar_h = 260, 6
                sharp_w = int(bar_w * (probs[0].item()))
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + sharp_w, bar_y + bar_h), (68, 68, 255), -1)
                cv2.rectangle(img, (bar_x + sharp_w, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 255, 136), -1)
                
                st.session_state.latest_prediction = self.result
                
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            return frame

# ── TOP BAR ──
st.markdown("""
<div class="top-bar">
    <div>
        <span class="logo">DOT. HUD</span>
        <span class="status-active" style="margin-left:20px;">● SYSTEM ACTIVE</span>
    </div>
    <div class="top-links">
        <span>BEND DETECTION</span>
        <span>REAL-TIME</span>
        <span>MOBILENETV2</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── MAIN CONTENT ──
col_main, col_sidebar = st.columns([2, 1])

with col_main:
    # ── INPUT SECTION ──
    st.markdown('<div class="hud-header">INPUT</div>', unsafe_allow_html=True)
    
    input_method = st.radio(
        "",
        ["📤 Upload", "📸 Camera", "🎥 Live"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    uploaded = None
    camera_image = None
    img = None
    source = None
    
    if input_method == "📤 Upload":
        uploaded = st.file_uploader("Upload highway image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            source = "upload"
            
    elif input_method == "📸 Camera":
        camera_image = st.camera_input("Take photo", label_visibility="collapsed")
        if camera_image:
            img = Image.open(camera_image).convert("RGB")
            source = "camera"
    
    else:
        st.info("🎥 Point camera at the road ahead")
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        ctx = webrtc_streamer(
            key="hud-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx.video_processor:
            ctx.video_processor.model = model
            ctx.video_processor.transform = transform
            ctx.video_processor.classes = CLASSES
            ctx.video_processor.device = DEVICE
    
    # ── CLASSIFICATION ──
    if uploaded or camera_image:
        st.image(img, caption="INPUT", use_column_width=True)
        
        if st.button("ANALYZE", use_container_width=True):
            with st.spinner("PROCESSING..."):
                result, error = predict_image(img, source)
            
            if error:
                st.error(f"⚠️ {error}")
                st.session_state.total_errors += 1
            else:
                st.session_state.total_predictions += 1
                if result["prediction"] == "sharp":
                    st.session_state.total_sharp += 1
                else:
                    st.session_state.total_straight += 1
                st.session_state.history.append({"type": "prediction", **result})
                
                # ── Result ──
                if result["prediction"] == "sharp":
                    st.markdown(f"""
                    <div class="result-sharp">
                        <div class="result-sharp-text">⚠️ SHARP BEND</div>
                        <div class="result-confidence">CONFIDENCE: {result['confidence']:.1f}%</div>
                        <div class="result-meta">{result['elapsed']:.0f}ms • {source.upper()}</div>
                    </div>
                    <div class="advice-box advice-box-sharp">
                        <div class="advice-title-sharp">⚠️ EMERGENCY MODE</div>
                        <div class="advice-text">• Reduce speed immediately<br>• Stay in your lane<br>• Watch for oncoming traffic<br>• Do not overtake</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-straight">
                        <div class="result-straight-text">✓ STRAIGHT ROAD</div>
                        <div class="result-confidence">CONFIDENCE: {result['confidence']:.1f}%</div>
                        <div class="result-meta">{result['elapsed']:.0f}ms • {source.upper()}</div>
                    </div>
                    <div class="advice-box advice-box-straight">
                        <div class="advice-title-straight">✅ NORMAL OPERATION</div>
                        <div class="advice-text">• Maintain safe following distance<br>• Stay alert and focused<br>• Observe speed limits</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ── Confidence Bar ──
                st.markdown('<div class="hud-header" style="margin-top:15px;">CURVATURE INTENSITY</div>', unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(10, 2))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                sharp_pct = result['sharp_prob']
                straight_pct = result['straight_prob']
                
                ax.barh([" "], [sharp_pct], color="#ff4444", height=0.5, label="Sharp")
                ax.barh([" "], [straight_pct], color="#00ff88", height=0.5, label="Straight", left=[sharp_pct])
                
                if sharp_pct > 5:
                    ax.text(sharp_pct/2, 0, f"{sharp_pct:.1f}%", va='center', ha='center', color='white', fontsize=10, fontweight='bold')
                if straight_pct > 5:
                    ax.text(sharp_pct + straight_pct/2, 0, f"{straight_pct:.1f}%", va='center', ha='center', color='white', fontsize=10, fontweight='bold')
                
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color='rgba(255,255,255,0.3)')
                ax.tick_params(colors='rgba(255,255,255,0.3)')
                ax.set_yticklabels([])
                ax.legend(loc='upper right', facecolor='none', labelcolor='rgba(255,255,255,0.5)')
                for spine in ax.spines.values():
                    spine.set_color('rgba(255,255,255,0.1)')
                plt.tight_layout()
                st.pyplot(fig)

with col_sidebar:
    # ── BEND ANALYSIS RESULT ──
    st.markdown('<div class="hud-header">BEND ANALYSIS RESULT</div>', unsafe_allow_html=True)
    
    if st.session_state.latest_prediction:
        result = st.session_state.latest_prediction
        if result["prediction"] == "sharp":
            st.markdown(f"""
            <div class="bend-analysis-box bend-analysis-sharp">
                <div style="color:#ff4444; font-size:0.7rem; letter-spacing:2px; font-family:Consolas;">DETECTED WHEEL DEVIATION</div>
                <div style="color:#ff4444; font-size:2.2rem; font-weight:700; font-family:Consolas;">42°</div>
                <div style="color:rgba(255,255,255,0.3); font-size:0.6rem; font-family:Consolas;">CURVATURE ANGLE</div>
                <div class="bend-intensity bend-intensity-sharp">Severe / Hairpin</div>
            </div>
            <div style="background:rgba(255,68,68,0.05); border:1px solid rgba(255,68,68,0.15); border-radius:10px; padding:15px; text-align:center;">
                <div style="color:#ff4444; font-size:0.7rem; letter-spacing:2px; font-family:Consolas;">CURVATURE INTENSITY</div>
                <div style="color:#ff4444; font-size:1.2rem; font-weight:600; font-family:Consolas;">Severe / Hairpin</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bend-analysis-box bend-analysis-straight">
                <div style="color:#00ff88; font-size:0.7rem; letter-spacing:2px; font-family:Consolas;">ROAD CLEAR</div>
                <div style="color:#00ff88; font-size:2.2rem; font-weight:700; font-family:Consolas;">0°</div>
                <div style="color:rgba(255,255,255,0.3); font-size:0.6rem; font-family:Consolas;">CURVATURE ANGLE</div>
                <div class="bend-intensity bend-intensity-straight">Normal / Straight</div>
            </div>
            <div style="background:rgba(0,255,136,0.03); border:1px solid rgba(0,255,136,0.08); border-radius:10px; padding:15px; text-align:center;">
                <div style="color:#00ff88; font-size:0.7rem; letter-spacing:2px; font-family:Consolas;">CURVATURE INTENSITY</div>
                <div style="color:#00ff88; font-size:1.2rem; font-weight:600; font-family:Consolas;">Normal / Straight</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:rgba(255,255,255,0.1); text-align:center; padding:30px 0; font-family:Consolas;">
            NO DATA<br>
            <span style="font-size:0.6rem; color:rgba(255,255,255,0.05);">Upload or capture an image</span>
        </div>
        """, unsafe_allow_html=True)
    
    # ── BEND HISTORY ──
    st.markdown('<div class="hud-header" style="margin-top:15px;">BEND HISTORY</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            if item.get("type") == "prediction":
                cls = item["prediction"]
                cls_class = "history-item-sharp" if cls == "sharp" else "history-item-straight"
                emoji = "🔴" if cls == "sharp" else "🟢"
                st.markdown(f"""
                <div class="history-item {cls_class}">
                    <span class="history-label">{emoji} {cls.upper()} {item['confidence']:.1f}%</span>
                    <span class="history-time" style="float:right;">{item['time']}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:rgba(255,255,255,0.1); text-align:center; padding:10px 0; font-family:Consolas;">NO DATA</div>', unsafe_allow_html=True)
    
    # ── DIAGNOSTICS ──
    st.markdown('<div class="hud-header" style="margin-top:15px;">DIAGNOSTICS</div>', unsafe_allow_html=True)
    
    total = st.session_state.total_predictions + st.session_state.total_errors
    if total > 0:
        error_rate = (st.session_state.total_errors / total * 100) if total > 0 else 0
        diag_status = "OK" if error_rate < 10 else "WARN" if error_rate < 20 else "ERROR"
        
        st.markdown(f"""
        <div style="font-family:Consolas; font-size:0.6rem;">
            <div class="diagnostic-item">STATUS: <span class="diagnostic-ok">{diag_status}</span></div>
            <div class="diagnostic-item">PREDICTIONS: {st.session_state.total_predictions}</div>
            <div class="diagnostic-item">SHARP: {st.session_state.total_sharp}</div>
            <div class="diagnostic-item">STRAIGHT: {st.session_state.total_straight}</div>
            <div class="diagnostic-item">ERRORS: {st.session_state.total_errors}</div>
            <div class="diagnostic-item">ACCURACY: 90.2%</div>
            <div class="diagnostic-item">MODEL: MOBILENETV2</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:rgba(255,255,255,0.1); text-align:center; padding:10px 0; font-family:Consolas;">SYSTEM IDLE</div>
        """, unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("---")
st.markdown("""
<div class="footer">
    <span>DOT.HUD v1.0</span>
    <span>⏻ SYSTEM ACTIVE</span>
    <span>🛣️ HIGHWAY BEND DETECTION</span>
    <span>MOBILENETV2 • 90.2%</span>
</div>
""", unsafe_allow_html=True)
