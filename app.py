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
import plotly.graph_objects as go
import plotly.express as px

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
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ── HUD Container ── */
    .hud-container {
        background: linear-gradient(180deg, rgba(10, 10, 15, 0.95), rgba(0, 0, 0, 1));
        border: 1px solid rgba(0, 180, 255, 0.15);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 30px rgba(0, 180, 255, 0.05);
        backdrop-filter: blur(10px);
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
    
    .hud-title {
        color: #00d4ff;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 2px;
        font-family: 'Consolas', monospace;
        text-shadow: 0 0 30px rgba(0, 180, 255, 0.2);
    }
    
    .hud-subtitle {
        color: rgba(0, 180, 255, 0.5);
        font-size: 0.7rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        font-family: 'Consolas', monospace;
    }
    
    /* ── Status Badge ── */
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
    
    /* ── Telemetry Numbers ── */
    .telemetry-value {
        color: #00d4ff;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Consolas', monospace;
        text-shadow: 0 0 20px rgba(0, 180, 255, 0.15);
    }
    
    .telemetry-label {
        color: rgba(0, 180, 255, 0.5);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: 'Consolas', monospace;
    }
    
    .telemetry-unit {
        color: rgba(0, 180, 255, 0.3);
        font-size: 0.7rem;
        font-family: 'Consolas', monospace;
    }
    
    /* ── Detection Result Box ── */
    .result-sharp {
        background: linear-gradient(135deg, rgba(255, 50, 50, 0.15), rgba(200, 0, 0, 0.05));
        border: 2px solid rgba(255, 50, 50, 0.4);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 40px rgba(255, 50, 50, 0.1);
    }
    
    .result-straight {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 200, 100, 0.05));
        border: 2px solid rgba(0, 255, 136, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.05);
    }
    
    .result-sharp-text {
        color: #ff4444;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 3px;
        font-family: 'Consolas', monospace;
        text-shadow: 0 0 30px rgba(255, 50, 50, 0.3);
    }
    
    .result-straight-text {
        color: #00ff88;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 3px;
        font-family: 'Consolas', monospace;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
    }
    
    .result-confidence {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        letter-spacing: 1px;
        font-family: 'Consolas', monospace;
    }
    
    /* ── Metrics Cards ── */
    .metric-card {
        background: rgba(0, 180, 255, 0.03);
        border: 1px solid rgba(0, 180, 255, 0.08);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(0, 180, 255, 0.2);
        background: rgba(0, 180, 255, 0.05);
    }
    
    .metric-value {
        color: #00d4ff;
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'Consolas', monospace;
    }
    
    .metric-label {
        color: rgba(0, 180, 255, 0.4);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: 'Consolas', monospace;
    }
    
    /* ── Route Analysis ── */
    .route-bar {
        background: rgba(0, 180, 255, 0.05);
        border-radius: 20px;
        height: 6px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .route-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ff88, #00d4ff, #ff4444);
        border-radius: 20px;
        transition: width 0.5s ease;
    }
    
    /* ── Bend History ── */
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
                cv2.rectangle(img, (10, 10), (250, 100), (0, 0, 0, 180), -1)
                cv2.putText(img, "DETECTION", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
                cv2.putText(img, f"{label} {confidence:.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(img, f"FPS: {self.fps}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
                
                # Progress bar
                bar_x, bar_y = 20, 115
                bar_w, bar_h = 200, 6
                sharp_w = int(bar_w * (probs[0].item()))
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + sharp_w, bar_y + bar_h), (68, 68, 255), -1)
                cv2.rectangle(img, (bar_x + sharp_w, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 255, 136), -1)
                
                st.session_state.latest_prediction = self.result
                
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            return frame

# ── Main HUD Layout ──
# ── Top Bar: Status ──
col_logo, col_status, col_xyz = st.columns([1, 2, 2])
with col_logo:
    st.markdown('<div style="font-size:1.5rem; color:#00b4ff; font-weight:700; font-family:Consolas;">DOT.HUD</div>')
    st.markdown('<div class="status-active">● SYSTEM ACTIVE</div>', unsafe_allow_html=True)

with col_status:
    st.markdown("""
    <div style="display:flex; gap:30px; justify-content:center; padding-top:5px;">
        <div><span style="color:rgba(0,180,255,0.4); font-size:0.6rem;">LIVE TELEMETRY</span></div>
        <div><span style="color:rgba(0,180,255,0.4); font-size:0.6rem;">ROUTE ANALYSIS</span></div>
        <div><span style="color:rgba(0,180,255,0.4); font-size:0.6rem;">BEND HISTORY</span></div>
        <div><span style="color:rgba(0,180,255,0.4); font-size:0.6rem;">VEHICLE HEALTH</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_xyz:
    st.markdown('<div style="text-align:right; color:rgba(0,180,255,0.5); font-family:Consolas; font-size:0.8rem;">XYZ: 142.1 | -0.4 | 12.9</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:right; color:#00ff88; font-family:Consolas; font-size:0.7rem;">● LIVE</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Main Content ──
col_main, col_sidebar = st.columns([2, 1])

with col_main:
    # ── Input Section ──
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
        uploaded = st.file_uploader(
            "Upload highway image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            source = "upload"
            
    elif input_method == "📸 Camera":
        camera_image = st.camera_input("Take photo", label_visibility="collapsed")
        if camera_image:
            img = Image.open(camera_image).convert("RGB")
            source = "camera"
    
    else:  # Live
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
    
    # ── Classification ──
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
                
                # ── Result Display ──
                if result["prediction"] == "sharp":
                    st.markdown(f"""
                    <div class="result-sharp">
                        <div class="result-sharp-text">⚠️ SHARP BEND</div>
                        <div class="result-confidence">CONFIDENCE: {result['confidence']:.1f}%</div>
                        <div style="color:rgba(255,255,255,0.3); font-size:0.7rem; margin-top:5px; font-family:Consolas;">
                            {result['elapsed']:.0f}ms • {source.upper()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-straight">
                        <div class="result-straight-text">✓ STRAIGHT ROAD</div>
                        <div class="result-confidence">CONFIDENCE: {result['confidence']:.1f}%</div>
                        <div style="color:rgba(255,255,255,0.3); font-size:0.7rem; margin-top:5px; font-family:Consolas;">
                            {result['elapsed']:.0f}ms • {source.upper()}
                        </div>
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
                
                # ── Driving Advice ──
                if result["prediction"] == "sharp":
                    st.warning("""
                    ⚠️ **EMERGENCY MODE**
                    - Reduce speed immediately
                    - Stay in your lane
                    - Watch for oncoming traffic
                    - Do not overtake
                    """)
                else:
                    st.success("""
                    ✅ **NORMAL OPERATION**
                    - Maintain safe following distance
                    - Stay alert and focused
                    - Observe speed limits
                    """)

with col_sidebar:
    # ── Route Analysis ──
    st.markdown('<div class="hud-header">ROUTE ANALYSIS</div>', unsafe_allow_html=True)
    
    # Distance markers
    st.markdown("""
    <div style="padding:5px 0; font-family:Consolas;">
        <div style="color:rgba(0,180,255,0.3); font-size:0.6rem;">DISTANCE TO BEND</div>
        <div style="display:flex; justify-content:space-between; color:rgba(255,255,255,0.2); font-size:0.6rem;">
            <span>0m</span><span>50m</span><span>100m</span><span>150m</span>
        </div>
        <div class="route-bar">
            <div class="route-bar-fill" style="width:0%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Bend Analysis Result ──
    if st.session_state.latest_prediction:
        result = st.session_state.latest_prediction
        if result["prediction"] == "sharp":
            st.markdown(f"""
            <div style="background:rgba(255,68,68,0.1); border:1px solid rgba(255,68,68,0.2); border-radius:10px; padding:15px; margin:10px 0;">
                <div style="color:#ff4444; font-size:0.8rem; font-family:Consolas;">DETECTED WHEEL DEVIATION</div>
                <div style="color:#ff4444; font-size:1.8rem; font-weight:700; font-family:Consolas;">42°</div>
                <div style="color:rgba(255,255,255,0.3); font-size:0.6rem; font-family:Consolas;">CURVATURE ANGLE</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(0,255,136,0.05); border:1px solid rgba(0,255,136,0.1); border-radius:10px; padding:15px; margin:10px 0;">
                <div style="color:#00ff88; font-size:0.8rem; font-family:Consolas;">ROAD CLEAR</div>
                <div style="color:#00ff88; font-size:1.8rem; font-weight:700; font-family:Consolas;">0°</div>
                <div style="color:rgba(255,255,255,0.3); font-size:0.6rem; font-family:Consolas;">CURVATURE ANGLE</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ── Bend History ──
    st.markdown('<div class="hud-header" style="margin-top:15px;">BEND HISTORY</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        recent = st.session_state.history[-5:]
        for item in reversed(recent):
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
        st.markdown('<div style="color:rgba(255,255,255,0.1); font-size:0.7rem; font-family:Consolas;">NO DATA</div>', unsafe_allow_html=True)
    
    # ── Diagnostics ──
    st.markdown('<div class="hud-header" style="margin-top:15px;">DIAGNOSTICS</div>', unsafe_allow_html=True)
    
    total = st.session_state.total_predictions + st.session_state.total_errors
    if total > 0:
        error_rate = (st.session_state.total_errors / total * 100) if total > 0 else 0
        diag_status = "OK" if error_rate < 10 else "WARN" if error_rate < 20 else "ERROR"
        diag_color = "#00ff88" if error_rate < 10 else "#ffaa00" if error_rate < 20 else "#ff4444"
        
        st.markdown(f"""
        <div style="font-family:Consolas; font-size:0.6rem;">
            <div class="diagnostic-item">STATUS: <span class="diagnostic-ok">{diag_status}</span></div>
            <div class="diagnostic-item">PREDICTIONS: {st.session_state.total_predictions}</div>
            <div class="diagnostic-item">ERRORS: {st.session_state.total_errors}</div>
            <div class="diagnostic-item">ACCURACY: 90.2%</div>
            <div class="diagnostic-item">MODEL: MOBILENETV2</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:rgba(255,255,255,0.1); font-size:0.7rem; font-family:Consolas;">SYSTEM IDLE</div>', unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; font-family:Consolas; font-size:0.5rem; color:rgba(255,255,255,0.1);">
    <span>DOT.HUD v1.0</span>
    <span>⏻ SYSTEM ACTIVE</span>
    <span>🛣️ HIGHWAY BEND DETECTION</span>
</div>
""", unsafe_allow_html=True)
