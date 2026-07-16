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

# ── Page Configuration ──
st.set_page_config(
    page_title="Highway Bend Classifier",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Professional CSS ──
st.markdown("""
<style>
    /* ── Main Background ── */
    .stApp {
        background: #0a0a0f;
    }
    
    /* ── Hide Default Elements ── */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ── Professional Header ── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        padding: 15px 0 5px 0;
        font-family: 'Segoe UI', sans-serif;
        letter-spacing: 1px;
    }
    
    .sub-header {
        color: rgba(255,255,255,0.4);
        text-align: center;
        font-size: 0.9rem;
        margin-bottom: 25px;
        font-family: 'Segoe UI', sans-serif;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    /* ── Cards ── */
    .card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: rgba(102, 126, 234, 0.2);
    }
    
    /* ── Prediction Boxes ── */
    .sharp-box {
        background: linear-gradient(135deg, rgba(255, 50, 50, 0.15), rgba(200, 0, 0, 0.05));
        border: 2px solid rgba(255, 50, 50, 0.3);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 30px rgba(255, 50, 50, 0.1);
    }
    
    .straight-box {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 200, 100, 0.05));
        border: 2px solid rgba(0, 255, 136, 0.25);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 30px rgba(0, 255, 136, 0.05);
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(255, 50, 50, 0.15), rgba(200, 0, 0, 0.05));
        border: 2px solid rgba(255, 50, 50, 0.3);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: white;
    }
    
    .sharp-text {
        color: #ff4444;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 2px;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .straight-text {
        color: #00ff88;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 2px;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .confidence-text {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        margin-top: 8px;
    }
    
    .meta-text {
        color: rgba(255,255,255,0.25);
        font-size: 0.7rem;
        margin-top: 5px;
    }
    
    /* ── Metrics ── */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.4);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 3px;
    }
    
    /* ── History Items ── */
    .history-item {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
        padding: 10px 15px;
        margin: 4px 0;
        border-left: 3px solid #667eea;
        transition: all 0.3s ease;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .history-item:hover {
        background: rgba(255, 255, 255, 0.05);
        transform: translateX(5px);
    }
    
    .history-item-sharp {
        border-left-color: #ff4444;
    }
    
    .history-item-straight {
        border-left-color: #00ff88;
    }
    
    .history-item-error {
        border-left-color: #ffaa00;
        background: rgba(255, 170, 0, 0.05);
    }
    
    .history-label {
        color: rgba(255,255,255,0.85);
        font-size: 0.85rem;
    }
    
    .history-time {
        color: rgba(255,255,255,0.25);
        font-size: 0.6rem;
        float: right;
    }
    
    .history-detail {
        color: rgba(255,255,255,0.3);
        font-size: 0.6rem;
    }
    
    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        font-size: 0.9rem;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* ── Upload Area ── */
    .upload-area {
        border: 1px dashed rgba(102, 126, 234, 0.2);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        background: rgba(255, 255, 255, 0.02);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: rgba(102, 126, 234, 0.4);
        background: rgba(255, 255, 255, 0.04);
    }
    
    /* ── Sidebar ── */
    .sidebar-title {
        color: #667eea;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
        padding: 10px 0;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .sidebar-sub {
        color: rgba(255,255,255,0.3);
        text-align: center;
        font-size: 0.7rem;
        margin-bottom: 20px;
        letter-spacing: 2px;
    }
    
    .stat-box {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }
    
    .stat-label {
        color: rgba(255,255,255,0.3);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* ── About Section ── */
    .about-text {
        color: rgba(255,255,255,0.7);
        line-height: 1.8;
    }
    
    .about-text strong {
        color: #667eea;
    }
    
    /* ── Footer ── */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.08);
        font-size: 0.5rem;
        padding: 20px 0 5px 0;
        font-family: 'Segoe UI', sans-serif;
        letter-spacing: 2px;
        border-top: 1px solid rgba(255,255,255,0.03);
        margin-top: 20px;
    }
    
    /* ── Radio Buttons ── */
    .stRadio > div {
        gap: 20px;
    }
    
    .stRadio label {
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.9rem !important;
    }
    
    .stRadio label[data-checked="true"] {
        color: #667eea !important;
    }
    
    /* ── Metrics Table ── */
    .metrics-table {
        width: 100%;
        color: rgba(255,255,255,0.7);
        font-size: 0.85rem;
    }
    
    .metrics-table td {
        padding: 6px 10px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .metrics-table .label {
        color: rgba(255,255,255,0.3);
    }
    
    .metrics-table .value {
        color: white;
        font-weight: 600;
        text-align: right;
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

# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛣️ Highway Bend</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Classifier</div>', unsafe_allow_html=True)
    
    page = st.radio(
        "",
        ["🏠 Home", "📊 Dashboard", "🕐 History", "ℹ️ About"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{st.session_state.total_predictions}</div>
            <div class="stat-label">Total</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color:#ff4444;">{st.session_state.total_sharp}</div>
            <div class="stat-label">Sharp</div>
        </div>
        """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color:#00ff88;">{st.session_state.total_straight}</div>
            <div class="stat-label">Straight</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color:#ffaa00;">{st.session_state.total_errors}</div>
            <div class="stat-label">Errors</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_sharp = 0
        st.session_state.total_straight = 0
        st.session_state.total_predictions = 0
        st.session_state.total_errors = 0
        st.rerun()

# ── Pages ──
if page == "🏠 Home":
    st.markdown('<div class="main-header">🛣️ Highway Bend Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning Road Safety System • 90.2% Accuracy</div>', unsafe_allow_html=True)
    
    input_method = st.radio(
        "Select input method:",
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
        
        if input_method == "📤 Upload Image":
            uploaded = st.file_uploader(
                "Upload a highway image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                help="Upload clear images of highways, roads, or streets"
            )
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, caption="Uploaded Image", use_column_width=True)
                source = "upload"
        else:
            camera_image = st.camera_input("Take a photo of the road ahead")
            if camera_image:
                img = Image.open(camera_image).convert("RGB")
                st.image(img, caption="Captured Image", use_column_width=True)
                source = "camera"
        
        if uploaded or camera_image:
            if st.button("🔍 Classify Image", use_container_width=True):
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
                        <div style="font-size:1.5rem;">❌ INVALID IMAGE</div>
                        <div style="font-size:1rem; margin-top:10px; opacity:0.8;">{error}</div>
                        <div style="font-size:0.8rem; margin-top:10px; opacity:0.5;">Please upload a clear image of a road or highway</div>
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
                            <div class="sharp-text">🔴 SHARP BEND DETECTED</div>
                            <div class="confidence-text">Confidence: {result['confidence']:.1f}%</div>
                            <div class="meta-text">⏱️ {result['elapsed']:.0f}ms • 📸 {source.title()}</div>
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
                            <div class="straight-text">🟢 STRAIGHT ROAD</div>
                            <div class="confidence-text">Confidence: {result['confidence']:.1f}%</div>
                            <div class="meta-text">⏱️ {result['elapsed']:.0f}ms • 📸 {source.title()}</div>
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
                    fig.patch.set_facecolor('#0a0a0f')
                    ax.set_facecolor('#0a0a0f')
                    
                    bars = ax.barh(CLASS_NAMES, 
                                  [result['sharp_prob'], result['straight_prob']],
                                  color=["#ff4444", "#00ff88"], height=0.5)
                    
                    for bar, val in zip(bars, [result['sharp_prob'], result['straight_prob']]):
                        ax.text(min(val + 2, 90), bar.get_y() + bar.get_height()/2,
                                f"{val:.1f}%", va='center', fontsize=14, 
                                fontweight='bold', color='white')
                    
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Confidence (%)", color='rgba(255,255,255,0.3)', fontsize=12)
                    ax.tick_params(colors='rgba(255,255,255,0.3)')
                    for spine in ax.spines.values():
                        spine.set_color('rgba(255,255,255,0.1)')
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
                    <div class="history-item history-item-error">
                        <span class="history-label">❌ Invalid Input</span>
                        <span class="history-time">{item['time']}</span>
                        <br>
                        <span class="history-detail">{item['error'][:50]}...</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    emoji = CLASS_EMOJIS[item["prediction"]]
                    cls_class = "history-item-sharp" if item["prediction"] == "sharp" else "history-item-straight"
                    st.markdown(f"""
                    <div class="history-item {cls_class}">
                        <span class="history-label">{emoji} {item['prediction'].upper()} {item['confidence']:.1f}%</span>
                        <span class="history-time">{item['time']}</span>
                        <br>
                        <span class="history-detail">📸 {item.get('source', 'upload').title()}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No activity yet. Upload or capture an image!")

elif page == "📊 Dashboard":
    st.markdown('<div class="main-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if st.session_state.total_predictions == 0 and st.session_state.total_errors == 0:
        st.info("📊 No data yet. Make some predictions to see analytics!")
        st.stop()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total", st.session_state.total_predictions + st.session_state.total_errors)
    with col2:
        st.metric("Sharp", st.session_state.total_sharp)
    with col3:
        st.metric("Straight", st.session_state.total_straight)
    with col4:
        st.metric("Errors", st.session_state.total_errors)
    with col5:
        error_rate = (st.session_state.total_errors / (st.session_state.total_predictions + st.session_state.total_errors) * 100) if (st.session_state.total_predictions + st.session_state.total_errors) > 0 else 0
        st.metric("Error Rate", f"{error_rate:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Class Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_facecolor('#0a0a0f')
        ax.pie([st.session_state.total_sharp, st.session_state.total_straight],
               labels=["Sharp Bend", "Straight Road"],
               colors=["#ff4444", "#00ff88"],
               autopct='%1.1f%%',
               startangle=90,
               textprops={'color': 'white', 'fontsize': 12})
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 📈 Confidence Trend")
        predictions = [h for h in st.session_state.history if h.get("type") == "prediction"]
        if predictions:
            df_history = pd.DataFrame(predictions)
            df_history['index'] = range(1, len(df_history) + 1)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0a0a0f')
            ax.set_facecolor('#0a0a0f')
            colors = ['#ff4444' if p == 'sharp' else '#00ff88' for p in df_history['prediction']]
            ax.scatter(df_history['index'], df_history['confidence'], c=colors, s=100, alpha=0.6)
            ax.plot(df_history['index'], df_history['confidence'], color='white', alpha=0.3)
            avg_conf = np.mean(df_history['confidence'])
            ax.axhline(y=avg_conf, color='#667eea', linestyle='--', label=f'Avg: {avg_conf:.1f}%')
            ax.set_xlabel('Prediction Number', color='rgba(255,255,255,0.3)')
            ax.set_ylabel('Confidence (%)', color='rgba(255,255,255,0.3)')
            ax.set_ylim(0, 105)
            ax.tick_params(colors='rgba(255,255,255,0.3)')
            for spine in ax.spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            ax.legend(facecolor='#0a0a0f', labelcolor='rgba(255,255,255,0.5)')
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("### 📋 Full History")
    if st.session_state.history:
        df_full = pd.DataFrame(st.session_state.history)
        df_full_display = df_full.copy()
        df_full_display['type'] = df_full_display.get('type', 'prediction')
        df_full_display['Class'] = df_full_display.apply(
            lambda x: x['prediction'].upper() if x.get('type') == 'prediction' else '❌ ERROR', 
            axis=1
        )
        df_full_display['Confidence %'] = df_full_display.apply(
            lambda x: f"{x['confidence']:.1f}%" if x.get('type') == 'prediction' else 'N/A',
            axis=1
        )
        df_full_display = df_full_display[['date', 'time', 'Class', 'Confidence %']]
        st.dataframe(df_full_display, use_container_width=True)
        
        csv = df_full.to_csv(index=False)
        st.download_button(
            label="📥 Download Full History CSV",
            data=csv,
            file_name="predictions_history.csv",
            mime="text/csv"
        )

elif page == "🕐 History":
    st.markdown('<div class="main-header">🕐 Prediction History</div>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("📭 No predictions yet. Upload or capture an image to get started!")
    else:
        filter_type = st.selectbox(
            "Filter by:",
            ["All", "Predictions Only", "Errors Only"]
        )
        
        filtered_history = st.session_state.history
        if filter_type == "Predictions Only":
            filtered_history = [h for h in st.session_state.history if h.get("type") == "prediction"]
        elif filter_type == "Errors Only":
            filtered_history = [h for h in st.session_state.history if h.get("type") == "error"]
        
        for i, item in enumerate(reversed(filtered_history)):
            if item.get("type") == "error":
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"### #{len(st.session_state.history) - i}")
                        st.caption(item["date"])
                    with col2:
                        st.markdown(f"""
                        <div class="history-item history-item-error">
                            <span class="history-label">❌ INVALID INPUT</span>
                            <span class="history-time">{item['time']}</span>
                            <br>
                            <span class="history-detail">Error: {item['error']}</span>
                            <br>
                            <span class="history-detail">📸 {item.get('source', 'upload').title()}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                emoji = CLASS_EMOJIS[item["prediction"]]
                cls_class = "history-item-sharp" if item["prediction"] == "sharp" else "history-item-straight"
                
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"### #{len(st.session_state.history) - i}")
                        st.caption(item["date"])
                    with col2:
                        st.markdown(f"""
                        <div class="history-item {cls_class}">
                            <span class="history-label">{emoji} {item['prediction'].upper()} - {item['confidence']:.1f}%</span>
                            <span class="history-time">{item['time']}</span>
                            <br>
                            <span class="history-detail">⏱️ {item['elapsed']:.0f}ms • 📸 {item.get('source', 'upload').title()}</span>
                            <br>
                            <span class="history-detail">Sharp: {item['sharp_prob']:.1f}% • Straight: {item['straight_prob']:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)

else:  # About
    st.markdown('<div class="main-header">ℹ️ About</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-text">
    ## 🛣️ Highway Bend Classifier
    
    ### Project Overview
    This application uses deep learning to classify highway images as either <strong>Sharp Bend</strong> or <strong>Straight Road</strong>.
    
    ### 🎯 Key Features
    - <strong>Multiple Input Methods:</strong> Upload images or take photos with your camera
    - <strong>Smart Validation:</strong> Automatically detects if the image contains a road
    - <strong>Real-time Classification:</strong> Instant predictions with confidence scores
    - <strong>History Tracking:</strong> Full prediction history with timestamps
    - <strong>Analytics Dashboard:</strong> View trends and statistics
    
    ### 📊 Model Performance
    <table class="metrics-table">
        <tr><td class="label">Architecture</td><td class="value">MobileNetV2</td></tr>
        <tr><td class="label">Accuracy</td><td class="value">90.2%</td></tr>
        <tr><td class="label">Sharp Precision</td><td class="value">92%</td></tr>
        <tr><td class="label">Sharp Recall</td><td class="value">89%</td></tr>
        <tr><td class="label">Straight Precision</td><td class="value">89%</td></tr>
        <tr><td class="label">Straight Recall</td><td class="value">92%</td></tr>
    </table>
    
    ### 🛠️ Technical Stack
    - <strong>Framework:</strong> PyTorch
    - <strong>Deployment:</strong> Streamlit Cloud
    - <strong>UI:</strong> Custom CSS + Streamlit
    - <strong>Computer Vision:</strong> MobileNetV2
    
    ### 📝 Important Note
    This is a research prototype. Always rely on official traffic signs and real-time conditions while driving.
    
    ### 🔗 Links
    - <a href="https://github.com/Oreoluwa03/highway-bend-classifier" style="color:#667eea;">Source Code</a>
    - <a href="https://huggingface.co/spaces/Oreoluwa82/Sharp-bend-detection" style="color:#667eea;">Hugging Face Space</a>
    
    ---
    <em style="color:rgba(255,255,255,0.3);">Built with ❤️ for road safety research</em>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class="footer">
    🛣️ Highway Bend Classifier • MobileNetV2 • 90.2% Accuracy • Built with ❤️
</div>
""", unsafe_allow_html=True)
