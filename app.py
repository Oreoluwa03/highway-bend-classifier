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
    page_title="Highway Bend Classifier",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
    }
    
    .sub-header {
        color: #a8b5d9;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 20px;
    }
    
    .sharp-box {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(238, 90, 36, 0.3);
    }
    
    .straight-box {
        background: linear-gradient(135deg, #00b894, #00a86b);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 168, 107, 0.3);
    }
    
    .error-box {
        background: linear-gradient(135deg, #ff6b6b, #c0392b);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(192, 57, 43, 0.3);
    }
    
    .history-item {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 3px 0;
        border-left: 3px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateX(5px);
    }
    
    .history-item-error {
        background: rgba(255, 0, 0, 0.05);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 3px 0;
        border-left: 3px solid #ff6b6b;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-label {
        color: #a8b5d9;
        font-size: 0.8rem;
    }
    
    .mode-selector {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .mode-selector label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──
CLASSES = ["sharp", "straight"]
CLASS_NAMES = ["Sharp Bend", "Straight Road"]
CLASS_COLORS = {"sharp": "#ff6b6b", "straight": "#00b894"}
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
        return False, "Image is too small. Please upload a larger image."
    
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
                if abs(r_val - g_val) < 30 and abs(g_val - b_val) < 30 and abs(r_val - b_val) < 30:
                    road_colors += 1
                elif r_val > 100 and g_val > 80 and b_val < 100:
                    road_colors += 1
                elif r_val > 150 and g_val > 150 and b_val > 150:
                    road_colors += 1
    
    road_ratio = road_colors / sample_size
    
    if road_ratio < 0.05:
        return False, "No road-like colors detected. Please upload a highway or road image."
    
    return True, "Valid road image"

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
        self.prediction_count = 0
        self.last_prediction_time = 0
        
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
                
                # Draw on frame
                label = f"{self.classes[pred].upper()}"
                color = (0, 255, 0) if self.classes[pred] == "straight" else (0, 0, 255)
                cv2.putText(img, f"{label} ({confidence:.1f}%)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(img, f"FPS: {self.fps}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Confidence bar
                bar_width = 200
                bar_height = 20
                sharp_width = int(bar_width * (probs[0].item()))
                cv2.rectangle(img, (10, 100), (10 + bar_width, 100 + bar_height), 
                             (50, 50, 50), -1)
                cv2.rectangle(img, (10, 100), (10 + sharp_width, 100 + bar_height), 
                             (0, 0, 255), -1)
                cv2.rectangle(img, (10 + sharp_width, 100), 
                             (10 + bar_width, 100 + bar_height), 
                             (0, 255, 0), -1)
                cv2.putText(img, f"Sharp: {probs[0].item()*100:.1f}%", (10, 135), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Update session state with latest prediction
                st.session_state.latest_prediction = self.result
                
                # Auto-save to history (only if confidence > 60% and different from last)
                current_time = time.time()
                if confidence > 60 and (current_time - self.last_prediction_time > 0.5):
                    # Check if different from last prediction
                    if not st.session_state.history or st.session_state.history[-1]["prediction"] != self.classes[pred]:
                        self.last_prediction_time = current_time
                        # We'll save in the main app to avoid session state conflicts
                
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            return frame

# ── Sidebar ──
with st.sidebar:
    st.markdown("# 🛣️ Highway Bend")
    st.markdown("### Classifier")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["📸 Home", "🎥 Live Feed", "📊 Dashboard", "🕐 History", "ℹ️ About"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", st.session_state.total_predictions)
    with col2:
        st.metric("Sharp", st.session_state.total_sharp)
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Straight", st.session_state.total_straight)
    with col4:
        avg_conf = np.mean([h["confidence"] for h in st.session_state.history]) if st.session_state.history else 0
        st.metric("Avg Conf", f"{avg_conf:.1f}%")
    
    if st.session_state.total_errors > 0:
        st.metric("⚠️ Errors", st.session_state.total_errors)
    
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_sharp = 0
        st.session_state.total_straight = 0
        st.session_state.total_predictions = 0
        st.session_state.total_errors = 0
        st.rerun()

# ── Pages ──
if page == "📸 Home":
    st.markdown('<div class="main-header">🛣️ Highway Bend Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning Road Safety System • 90.2% Accuracy</div>', unsafe_allow_html=True)
    
    # Mode Selection
    st.markdown("### 📌 Select Input Mode")
    mode = st.radio(
        "",
        ["📤 Upload Image", "📸 Take Photo"],
        horizontal=True
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Input")
        
        uploaded = None
        camera_image = None
        img = None
        source = None
        
        if mode == "📤 Upload Image":
            uploaded = st.file_uploader(
                "Upload a highway image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                help="Upload clear images of highways, roads, or streets"
            )
            
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, caption="Uploaded Image", use_column_width=True)
                source = "upload"
        
        else:  # Camera
            camera_image = st.camera_input("Take a photo of the road ahead")
            
            if camera_image:
                img = Image.open(camera_image).convert("RGB")
                st.image(img, caption="Captured Image", use_column_width=True)
                source = "camera"
        
        # Classify button
        if uploaded or camera_image:
            if st.button("🔍 Classify Image", use_container_width=True, type="primary"):
                with st.spinner("🔄 Analyzing image..."):
                    result, error = predict_image(img, source)
                
                if error:
                    st.session_state.total_errors += 1
                    st.session_state.history.append({
                        "type": "error",
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "error": error,
                        "source": source
                    })
                    
                    st.markdown(f"""
                    <div class="error-box">
                        <h2 style="margin:0;">❌ INVALID IMAGE</h2>
                        <p style="font-size:1rem; margin:10px 0;">{error}</p>
                        <p style="opacity:0.8;">Please upload a clear image of a road or highway</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning("""
                    📸 **Tips for better results:**
                    - Take a photo of the road ahead
                    - Ensure good lighting
                    - Frame the road clearly
                    - Avoid photos with people or buildings
                    """)
                    
                else:
                    st.session_state.total_predictions += 1
                    if result["prediction"] == "sharp":
                        st.session_state.total_sharp += 1
                    else:
                        st.session_state.total_straight += 1
                    
                    st.session_state.history.append({
                        "type": "prediction",
                        **result
                    })
                    
                    if result["prediction"] == "sharp":
                        st.markdown(f"""
                        <div class="sharp-box">
                            <h2 style="margin:0;">🔴 SHARP BEND DETECTED</h2>
                            <p style="font-size:1.1rem; margin:10px 0;">Confidence: {result['confidence']:.1f}%</p>
                            <p style="opacity:0.8;">⏱️ Processed in {result['elapsed']:.0f}ms • 📸 {source.title()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.warning("""
                        ⚠️ **Driving Advice — Sharp Bend**
                        - Reduce speed immediately
                        - Stay in your lane
                        - Watch for oncoming traffic
                        - Do not overtake
                        - Use headlights in low visibility
                        """)
                    else:
                        st.markdown(f"""
                        <div class="straight-box">
                            <h2 style="margin:0;">🟢 STRAIGHT ROAD</h2>
                            <p style="font-size:1.1rem; margin:10px 0;">Confidence: {result['confidence']:.1f}%</p>
                            <p style="opacity:0.8;">⏱️ Processed in {result['elapsed']:.0f}ms • 📸 {source.title()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success("""
                        ✅ **Driving Advice — Straight Road**
                        - Normal driving conditions
                        - Maintain safe following distance
                        - Stay alert and focused
                        - Observe speed limits
                        """)
                    
                    # Confidence Chart
                    st.markdown("### 📊 Confidence Analysis")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    fig.patch.set_facecolor('#1a1a2e')
                    ax.set_facecolor('#1a1a2e')
                    
                    bars = ax.barh(CLASS_NAMES, 
                                  [result['sharp_prob'], result['straight_prob']],
                                  color=["#ff6b6b", "#00b894"], height=0.5)
                    
                    for bar, val in zip(bars, [result['sharp_prob'], result['straight_prob']]):
                        ax.text(min(val + 2, 90), bar.get_y() + bar.get_height()/2,
                                f"{val:.1f}%", va='center', fontsize=14, 
                                fontweight='bold', color='white')
                    
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Confidence (%)", color='white', fontsize=12)
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values():
                        spine.set_color('white')
                    plt.tight_layout()
                    st.pyplot(fig)
    
    with col2:
        st.markdown("### 📈 Performance Summary")
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">90.2%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            avg_conf = np.mean([h["confidence"] for h in st.session_state.history 
                              if h.get("type") == "prediction"]) if st.session_state.history else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_conf:.1f}%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Model Performance")
        metrics_data = {
            "Class": ["Sharp Bend", "Straight Road"],
            "Precision": ["92%", "89%"],
            "Recall": ["89%", "92%"],
            "F1-Score": ["90%", "90%"]
        }
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, hide_index=True, use_container_width=True)
        
        st.markdown("### 📈 Recent Activity")
        if st.session_state.history:
            recent = st.session_state.history[-3:]
            for item in reversed(recent):
                if item.get("type") == "error":
                    st.markdown(f"""
                    <div class="history-item-error">
                        <b>❌ Invalid Input</b>
                        <span style="float:right; color:#a8b5d9;">{item['time']}</span>
                        <br>
                        <small style="color:#a8b5d9;">{item['error'][:50]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    emoji = CLASS_EMOJIS[item["prediction"]]
                    color = "#ff6b6b" if item["prediction"] == "sharp" else "#00b894"
                    st.markdown(f"""
                    <div class="history-item" style="border-left-color: {color};">
                        <b>{emoji} {item['prediction'].upper()}</b>
                        <span style="float:right; color:#a8b5d9;">{item['confidence']:.1f}%</span>
                        <br>
                        <small style="color:#a8b5d9;">🕐 {item['time']} • 📸 {item.get('source', 'upload').title()}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No activity yet. Upload or capture an image!")

elif page == "🎥 Live Feed":
    st.markdown('<div class="main-header">🛣️ Live Highway Bend Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🔴 Sharp Bend Detection • 🟢 Straight Road Detection • Real-time Analysis</div>', unsafe_allow_html=True)
    
    # Live feed controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 🎥 Camera Feed")
        st.caption("Enable camera and point at the road ahead")
    
    with col2:
        st.markdown("### 📊 Detection Mode")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
    
    with col3:
        st.markdown("### ⚙️ Auto-Save")
        auto_save = st.checkbox("Auto-save predictions", value=True)
    
    # Live video stream
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    ctx = webrtc_streamer(
        key="highway-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Set up the video processor with model
    if ctx.video_processor:
        ctx.video_processor.model = model
        ctx.video_processor.transform = transform
        ctx.video_processor.classes = CLASSES
        ctx.video_processor.device = DEVICE
    
    # Display current prediction and auto-save
    if st.session_state.latest_prediction:
        result = st.session_state.latest_prediction
        
        # Auto-save to history
        if auto_save and result["confidence"] >= confidence_threshold * 100:
            if not st.session_state.history or st.session_state.history[-1]["prediction"] != result["prediction"]:
                st.session_state.total_predictions += 1
                if result["prediction"] == "sharp":
                    st.session_state.total_sharp += 1
                else:
                    st.session_state.total_straight += 1
                
                st.session_state.history.append({
                    "type": "prediction",
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "sharp_prob": result["sharp_prob"],
                    "straight_prob": result["straight_prob"],
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "live"
                })
        
        # Display result
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if result["prediction"] == "sharp":
                st.markdown(f"""
                <div class="sharp-box">
                    <h2>🔴 SHARP BEND</h2>
                    <p style="font-size:1.2rem;">Confidence: {result['confidence']:.1f}%</p>
                    <p style="opacity:0.8;">⚠️ Reduce speed immediately!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="straight-box">
                    <h2>🟢 STRAIGHT ROAD</h2>
                    <p style="font-size:1.2rem;">Confidence: {result['confidence']:.1f}%</p>
                    <p style="opacity:0.8;">✅ Normal driving conditions</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            
            bars = ax.barh(CLASS_NAMES, 
                          [result['sharp_prob'], result['straight_prob']],
                          color=["#ff6b6b", "#00b894"], height=0.5)
            
            for bar, val in zip(bars, [result['sharp_prob'], result['straight_prob']]):
                ax.text(min(val + 2, 90), bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va='center', fontsize=12, 
                        fontweight='bold', color='white')
            
            ax.set_xlim(0, 100)
            ax.set_xlabel("Confidence (%)", color='white', fontsize=10)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            plt.tight_layout()
            st.pyplot(fig)
    
    else:
        st.info("📸 Please enable your camera and point it at the road ahead")
        st.caption("""
        **Tips for best results:**
        - Ensure good lighting
        - Point camera at the road ahead
        - Keep the road centered in frame
        - Avoid glare and reflections
        """)

elif page == "📊 Dashboard":
    st.markdown('<div class="main-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if st.session_state.total_predictions == 0:
        st.info("📊 No data yet. Make some predictions to see analytics!")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", st.session_state.total_predictions)
    with col2:
        st.metric("Sharp Bends", st.session_state.total_sharp)
    with col3:
        st.metric("Straight Roads", st.session_state.total_straight)
    with col4:
        avg_conf = np.mean([h["confidence"] for h in st.session_state.history])
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Class Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        ax.pie([st.session_state.total_sharp, st.session_state.total_straight],
               labels=["Sharp Bend", "Straight Road"],
               colors=["#ff6b6b", "#00b894"],
               autopct='%1.1f%%',
               startangle=90,
               textprops={'color': 'white', 'fontsize': 12})
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 📈 Confidence Trend")
        if st.session_state.history:
            df_history = pd.DataFrame(st.session_state.history)
            df_history['index'] = range(1, len(df_history) + 1)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            colors = ['#ff6b6b' if p == 'sharp' else '#00b894' for p in df_history['prediction']]
            ax.scatter(df_history['index'], df_history['confidence'], c=colors, s=100, alpha=0.6)
            ax.plot(df_history['index'], df_history['confidence'], color='white', alpha=0.3)
            avg_conf = np.mean(df_history['confidence'])
            ax.axhline(y=avg_conf, color='#667eea', linestyle='--', label=f'Avg: {avg_conf:.1f}%')
            ax.set_xlabel('Prediction Number', color='white')
            ax.set_ylabel('Confidence (%)', color='white')
            ax.set_ylim(0, 105)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.legend(facecolor='#1a1a2e', labelcolor='white')
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("### 📋 Full History")
    if st.session_state.history:
        df_full = pd.DataFrame(st.session_state.history)
        df_full['prediction'] = df_full['prediction'].str.upper()
        df_full = df_full[['date', 'time', 'prediction', 'confidence', 'sharp_prob', 'straight_prob']]
        df_full.columns = ['Date', 'Time', 'Class', 'Confidence %', 'Sharp %', 'Straight %']
        st.dataframe(df_full, use_container_width=True)
        
        csv = df_full.to_csv(index=False)
        st.download_button(
            label="📥 Download History CSV",
            data=csv,
            file_name="predictions_history.csv",
            mime="text/csv"
        )

elif page == "🕐 History":
    st.markdown('<div class="main-header">🕐 Prediction History</div>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("📭 No predictions yet. Upload an image or start the live feed!")
    else:
        filter_type = st.selectbox(
            "Filter by:",
            ["All", "Predictions Only", "Errors Only", "Live Feed", "Uploads"]
        )
        
        filtered_history = st.session_state.history
        if filter_type == "Predictions Only":
            filtered_history = [h for h in st.session_state.history if h.get("type") == "prediction"]
        elif filter_type == "Errors Only":
            filtered_history = [h for h in st.session_state.history if h.get("type") == "error"]
        elif filter_type == "Live Feed":
            filtered_history = [h for h in st.session_state.history if h.get("source") == "live"]
        elif filter_type == "Uploads":
            filtered_history = [h for h in st.session_state.history if h.get("source") in ["upload", "camera"]]
        
        for i, item in enumerate(reversed(filtered_history)):
            if item.get("type") == "error":
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"### #{len(st.session_state.history) - i}")
                        st.caption(item["date"])
                    with col2:
                        st.markdown(f"""
                        <div class="history-item-error">
                            <h3 style="margin:0; color: #ff6b6b;">❌ INVALID INPUT</h3>
                            <p style="margin:5px 0;">
                                Error: <b>{item['error']}</b>
                            </p>
                            <p style="margin:5px 0; font-size:0.9rem; color:#a8b5d9;">
                                📸 {item.get('source', 'upload').title()} • 🕐 {item['time']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                emoji = CLASS_EMOJIS[item["prediction"]]
                color = "#ff6b6b" if item["prediction"] == "sharp" else "#00b894"
                source_icon = "📺" if item.get("source") == "live" else "📸"
                
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"### #{len(st.session_state.history) - i}")
                        st.caption(item["date"])
                    with col2:
                        st.markdown(f"""
                        <div class="history-item" style="border-left-color: {color};">
                            <h3 style="margin:0; color: {color};">{emoji} {item['prediction'].upper()}</h3>
                            <p style="margin:5px 0;">
                                Confidence: <b>{item['confidence']:.1f}%</b>
                                • Processed: <b>{item['elapsed']:.0f}ms</b>
                                • {source_icon} {item.get('source', 'upload').title()}
                            </p>
                            <p style="margin:5px 0; font-size:0.9rem; color:#a8b5d9;">
                                Sharp: {item['sharp_prob']:.1f}% • Straight: {item['straight_prob']:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

else:  # About
    st.markdown('<div class="main-header">ℹ️ About</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🛣️ Highway Bend Classifier
    
    ### 🎯 Features
    - **📸 Image Upload:** Upload highway images for classification
    - **📸 Camera Capture:** Take photos directly from your device
    - **🎥 Live Video Feed:** Real-time road detection
    - **🧠 Smart Validation:** Automatically validates road images
    - **📊 Analytics Dashboard:** Track predictions and trends
    - **🕐 History Tracking:** Full prediction history with filters
    
    ### 📊 Model Performance
    | Metric | Value |
    |--------|-------|
    | Architecture | MobileNetV2 |
    | Accuracy | 90.2% |
    | Sharp Precision | 92% |
    | Sharp Recall | 89% |
    | Straight Precision | 89% |
    | Straight Recall | 92% |
    
    ### 🎥 Live Feed Features
    - Real-time frame analysis
    - FPS counter
    - Visual overlays
    - Confidence bars
    - Auto-save predictions
    
    ### 🛠️ Technical Stack
    - **Framework:** PyTorch
    - **Deployment:** Streamlit Cloud
    - **Video:** WebRTC + OpenCV
    - **UI:** Custom CSS + Streamlit
    
    ### 📝 Important Note
    This is a research prototype. Always rely on official traffic signs and real-time conditions while driving.
    
    ### 🔗 Links
    - [Source Code](https://github.com/Oreoluwa03/highway-bend-classifier)
    - [Model Details](https://huggingface.co/spaces/Oreoluwa82/Sharp-bend-detection)
    
    ---
    *Built with ❤️ for road safety research*
    """)
