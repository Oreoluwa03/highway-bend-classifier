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
import io

# ── Page Configuration ──
st.set_page_config(
    page_title="Highway Bend Classifier",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Professional Look ──
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    
    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin: 10px 0;
    }
    
    /* Headers */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
    }
    
    .sub-header {
        color: #a8b5d9;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Prediction boxes */
    .sharp-box {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(238, 90, 36, 0.3);
    }
    
    .straight-box {
        background: linear-gradient(135deg, #00b894, #00a86b);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 168, 107, 0.3);
    }
    
    .error-box {
        background: linear-gradient(135deg, #ff6b6b, #c0392b);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(192, 57, 43, 0.3);
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-label {
        color: #a8b5d9;
        font-size: 0.9rem;
    }
    
    /* History items */
    .history-item {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px 0;
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
        padding: 10px 15px;
        margin: 5px 0;
        border-left: 3px solid #ff6b6b;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader */
    .upload-container {
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Camera button */
    .camera-btn {
        background: linear-gradient(135deg, #00b894, #00a86b) !important;
    }
    
    .camera-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 168, 107, 0.4);
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

# ── Input Validation Function ──
def validate_road_image(image):
    """
    Validate if the image contains a road/highway scene.
    Returns: (is_valid, reason)
    """
    # Convert to numpy for analysis
    img_array = np.array(image)
    
    # Check 1: Image dimensions (too small images are likely invalid)
    if img_array.shape[0] < 50 or img_array.shape[1] < 50:
        return False, "Image is too small. Please upload a larger image."
    
    # Check 2: Check for road-like colors (brown, gray, white)
    # Convert to HSV for better color analysis
    from colorsys import rgb_to_hsv
    
    # Sample pixels to check colors
    h, w = img_array.shape[:2]
    sample_size = min(1000, h * w // 10)
    sample_indices = np.random.choice(h * w, sample_size, replace=False)
    
    road_colors = 0
    for idx in sample_indices:
        r, c = divmod(idx, w)
        if r < h and c < w:
            pixel = img_array[r, c]
            # Check if pixel is gray/brown/white (road-like colors)
            if len(pixel) >= 3:
                r_val, g_val, b_val = pixel[:3]
                # Grayscale check
                if abs(r_val - g_val) < 30 and abs(g_val - b_val) < 30 and abs(r_val - b_val) < 30:
                    road_colors += 1
                # Brown/beige check
                elif r_val > 100 and g_val > 80 and b_val < 100:
                    road_colors += 1
                # White/light gray
                elif r_val > 150 and g_val > 150 and b_val > 150:
                    road_colors += 1
    
    road_ratio = road_colors / sample_size
    
    if road_ratio < 0.05:
        return False, "No road-like colors detected. Please upload a highway or road image."
    
    return True, "Valid road image"

# ── Prediction Function ──
def predict_image(img, source="upload"):
    """
    Predict image class with validation.
    Returns: (result_dict, error_message)
    """
    # Validate input
    is_valid, reason = validate_road_image(img)
    if not is_valid:
        return None, reason
    
    # Predict
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

# ── Sidebar Navigation ──
with st.sidebar:
    st.markdown("# 🛣️ Highway Bend")
    st.markdown("### Classifier")
    st.markdown("---")
    
    # Simple navigation with buttons
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Dashboard", "🕐 History", "ℹ️ About"],
        index=0
    )
    
    st.markdown("---")
    
    # Sidebar stats
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", st.session_state.total_predictions, delta=None)
    with col2:
        st.metric("Sharp", st.session_state.total_sharp, delta=None)
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Straight", st.session_state.total_straight, delta=None)
    with col4:
        st.metric("⚠️ Errors", st.session_state.total_errors, delta=None)
    
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_sharp = 0
        st.session_state.total_straight = 0
        st.session_state.total_predictions = 0
        st.session_state.total_errors = 0
        st.rerun()

# ── Page Content ──
if page == "🏠 Home":
    st.markdown('<div class="main-header">🛣️ Highway Bend Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning Road Safety System • 90.2% Accuracy</div>', unsafe_allow_html=True)
    
    # Input method selection
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
                    # ── Invalid Input ──
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
                        <h1 style="margin:0;">❌ INVALID IMAGE</h1>
                        <p style="font-size:1.1rem; margin:10px 0;">{error}</p>
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
                    # ── Valid Prediction ──
                    st.session_state.total_predictions += 1
                    if result["prediction"] == "sharp":
                        st.session_state.total_sharp += 1
                    else:
                        st.session_state.total_straight += 1
                    
                    st.session_state.history.append({
                        "type": "prediction",
                        **result
                    })
                    
                    # Display result
                    if result["prediction"] == "sharp":
                        st.markdown(f"""
                        <div class="sharp-box">
                            <h1 style="margin:0;">🔴 SHARP BEND DETECTED</h1>
                            <p style="font-size:1.2rem; margin:10px 0;">Confidence: {result['confidence']:.1f}%</p>
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
                            <h1 style="margin:0;">🟢 STRAIGHT ROAD</h1>
                            <p style="font-size:1.2rem; margin:10px 0;">Confidence: {result['confidence']:.1f}%</p>
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
                    
                    # ── Confidence Chart ──
                    st.markdown("### 📊 Confidence Analysis")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    fig.patch.set_facecolor('transparent')
                    ax.set_facecolor('transparent')
                    
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
        
        # Metrics
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
        
        # Per-class metrics table
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

elif page == "📊 Dashboard":
    st.markdown('<div class="main-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if st.session_state.total_predictions == 0 and st.session_state.total_errors == 0:
        st.info("📊 No data yet. Make some predictions to see analytics!")
        st.stop()
    
    # Overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total", st.session_state.total_predictions + st.session_state.total_errors)
    with col2:
        st.metric("Sharp", st.session_state.total_sharp)
    with col3:
        st.metric("Straight", st.session_state.total_straight)
    with col4:
        st.metric("⚠️ Errors", st.session_state.total_errors)
    with col5:
        error_rate = (st.session_state.total_errors / (st.session_state.total_predictions + st.session_state.total_errors) * 100) if (st.session_state.total_predictions + st.session_state.total_errors) > 0 else 0
        st.metric("Error Rate", f"{error_rate:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Class Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
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
        predictions = [h for h in st.session_state.history if h.get("type") == "prediction"]
        if predictions:
            df_history = pd.DataFrame(predictions)
            df_history['index'] = range(1, len(df_history) + 1)
            
            fig, ax = plt.subplots(figsize=(10, 5))
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
        
        # Download button
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
        # Filter
        filter_type = st.selectbox(
            "Filter by:",
            ["All", "Predictions Only", "Errors Only"]
        )
        
        filtered_history = st.session_state.history
        if filter_type == "Predictions Only":
            filtered_history = [h for h in st.session_state.history if h.get("type") == "prediction"]
        elif filter_type == "Errors Only":
            filtered_history = [h for h in st.session_state.history if h.get("type") == "error"]
        
        # Timeline view
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
                                • 📸 {item.get('source', 'upload').title()}
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
    
    ### Project Overview
    This application uses deep learning to classify highway images as either **Sharp Bend** or **Straight Road**.
    
    ### 🎯 Key Features
    - **Multiple Input Methods:** Upload images or take photos with your camera
    - **Smart Validation:** Automatically detects if the image contains a road
    - **Real-time Classification:** Instant predictions with confidence scores
    - **History Tracking:** Full prediction history with timestamps
    - **Analytics Dashboard:** View trends and statistics
    
    ### 📊 Model Performance
    | Metric | Value |
    |--------|-------|
    | Architecture | MobileNetV2 |
    | Accuracy | 90.2% |
    | Sharp Precision | 92% |
    | Sharp Recall | 89% |
    | Straight Precision | 89% |
    | Straight Recall | 92% |
    
    ### 📸 Input Methods
    1. **Upload Image:** Upload a saved photo from your device
    2. **Take Photo:** Use your camera to capture a real-time image
    
    ### ✅ Validation Features
    - Checks if the image contains road-like colors
    - Ensures minimum image quality
    - Validates image dimensions
    - Provides clear error messages
    
    ### 🛠️ Technical Stack
    - **Framework:** PyTorch
    - **Deployment:** Streamlit Cloud
    - **UI:** Custom CSS + Streamlit
    - **Computer Vision:** MobileNetV2
    
    ### 📝 Important Note
    This is a research prototype. Always rely on official traffic signs and real-time conditions while driving.
    
    ### 🔗 Links
    - [Source Code](https://github.com/Oreoluwa03/highway-bend-classifier)
    - [Model Details](https://huggingface.co/spaces/Oreoluwa82/Sharp-bend-detection)
    
    ---
    *Built with ❤️ for road safety research*
    """)
