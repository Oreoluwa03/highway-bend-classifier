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
import asyncio

# ── Page Configuration ──
st.set_page_config(
    page_title="Highway Bend Classifier - Live",
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
    
    .live-badge {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
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
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
if "running" not in st.session_state:
    st.session_state.running = False

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
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(img_rgb)
            
            # Predict
            if self.model is not None:
                # Resize and transform
                tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    pred = probs.argmax().item()
                    confidence = probs[pred].item() * 100
                
                # Store result
                self.result = {
                    "prediction": self.classes[pred],
                    "confidence": confidence,
                    "sharp_prob": probs[0].item() * 100,
                    "straight_prob": probs[1].item() * 100
                }
                
                # Calculate FPS
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
                
                # Draw confidence bar
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
                
                # Update session state
                st.session_state.latest_prediction = self.result
                
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            return frame

# ── Sidebar ──
with st.sidebar:
    st.markdown("# 🛣️ Highway Bend")
    st.markdown("### Live Classifier")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigate",
        ["🎥 Live Feed", "📊 Dashboard", "🕐 History", "ℹ️ About"],
        index=0
    )
    
    st.markdown("---")
    
    # Stats
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
    
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_sharp = 0
        st.session_state.total_straight = 0
        st.session_state.total_predictions = 0
        st.rerun()

# ── Pages ──
if page == "🎥 Live Feed":
    st.markdown('<div class="main-header">🛣️ Live Highway Bend Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🔴 Sharp Bend Detection • 🟢 Straight Road Detection • Real-time Analysis</div>', unsafe_allow_html=True)
    
    # Live feed controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 🎥 Camera Feed")
        st.caption("Enable camera and point at the road ahead")
    
    with col2:
        st.markdown("### 📊 Detection Mode")
        mode = st.selectbox("", ["Real-time", "Sample Images"], label_visibility="collapsed")
    
    with col3:
        st.markdown("### ⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
    
    if mode == "Real-time":
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
        
        # Display current prediction
        if st.session_state.latest_prediction:
            result = st.session_state.latest_prediction
            
            # Save to history (only if confidence > threshold)
            if result["confidence"] >= confidence_threshold * 100:
                # Check if this is a new prediction (avoid duplicates)
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
                # Confidence chart
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
    
    else:  # Sample Images
        st.info("📸 Sample image mode coming soon!")
        st.caption("For now, use the Live Feed mode above")

elif page == "📊 Dashboard":
    st.markdown('<div class="main-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if st.session_state.total_predictions == 0:
        st.info("📊 No data yet. Start the live feed to collect data!")
        st.stop()
    
    # Overview metrics
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
    
    # Charts
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
    
    # Full history table
    st.markdown("### 📋 Full History")
    if st.session_state.history:
        df_full = pd.DataFrame(st.session_state.history)
        df_full['prediction'] = df_full['prediction'].str.upper()
        df_full = df_full[['date', 'time', 'prediction', 'confidence', 'sharp_prob', 'straight_prob']]
        df_full.columns = ['Date', 'Time', 'Class', 'Confidence %', 'Sharp %', 'Straight %']
        st.dataframe(df_full, use_container_width=True)
        
        # Download button
        csv = df_full.to_csv(index=False)
        st.download_button(
            label="📥 Download History CSV",
            data=csv,
            file_name="live_predictions_history.csv",
            mime="text/csv"
        )

elif page == "🕐 History":
    st.markdown('<div class="main-header">🕐 Prediction History</div>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("📭 No predictions yet. Start the live feed to collect data!")
    else:
        for i, item in enumerate(reversed(st.session_state.history[-50:])):
            emoji = CLASS_EMOJIS[item["prediction"]]
            color = "#ff6b6b" if item["prediction"] == "sharp" else "#00b894"
            
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"### #{len(st.session_state.history) - i}")
                    st.caption(item["date"])
                with col2:
                    st.markdown(f"""
                    <div class="history-item" style="border-left-color: {color};">
                        <b>{emoji} {item['prediction'].upper()}</b>
                        <span style="float:right; color:#a8b5d9;">{item['confidence']:.1f}%</span>
                        <br>
                        <small style="color:#a8b5d9;">
                            🕐 {item['time']} • 📺 Live Feed
                        </small>
                        <br>
                        <small style="color:#a8b5d9;">
                            Sharp: {item['sharp_prob']:.1f}% • Straight: {item['straight_prob']:.1f}%
                        </small>
                    </div>
                    """, unsafe_allow_html=True)

else:  # About
    st.markdown('<div class="main-header">ℹ️ About</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🛣️ Highway Bend Classifier - Live Edition
    
    ### 🎯 What's New
    - **Live Video Feed:** Real-time road detection using your camera
    - **FPS Display:** Real-time frame rate monitoring
    - **Visual Overlay:** Prediction displayed directly on video
    - **Confidence Bar:** Visual confidence indicator
    
    ### 📊 Live Features
    - **Instant Detection:** Analyze each frame in real-time
    - **History Tracking:** Record all predictions
    - **Confidence Threshold:** Adjustable sensitivity
    - **Performance Metrics:** Track FPS and accuracy
    
    ### 🛠️ Technical Details
    - **Backend:** PyTorch + OpenCV
    - **Frontend:** Streamlit + WebRTC
    - **Model:** MobileNetV2 (90.2% accuracy)
    - **Processing:** Real-time frame analysis
    
    ### 🎥 Usage Tips
    1. Enable camera access
    2. Point at the road ahead
    3. Keep camera stable
    4. Ensure good lighting
    
    ---
    *Built with ❤️ for road safety research*
    """)
