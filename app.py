
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import time

# ── Page Config ──
st.set_page_config(
    page_title="Highway Bend Classifier",
    page_icon="🛣️",
    layout="wide"
)

# ── Constants ──
CLASSES = ["sharp", "straight"]
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = load_model()

# ── Transform ──
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── UI ──
st.markdown("""
# 🛣️ Highway Bend Classifier
### Deep Learning Road Safety System — 90.2% Accuracy
Upload a highway image to classify it as a **Sharp Bend** or **Straight Road**
""")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload Highway Image", type=["jpg", "jpeg", "png"])
    
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Classify Image", type="primary"):
            with st.spinner("🔄 Analyzing image..."):
                start = time.time()
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = model(tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    pred = probs.argmax().item()
                
                elapsed = (time.time() - start) * 1000
                predicted_class = CLASSES[pred]
                confidence = probs[pred].item() * 100
            
            # Show result
            if predicted_class == "sharp":
                st.error(f"🔴 **SHARP BEND DETECTED**\nConfidence: {confidence:.1f}%\n⏱️ {elapsed:.0f}ms")
                st.warning("""
                ⚠️ **Driving Advice — Sharp Bend**
                - Reduce speed immediately
                - Stay in your lane
                - Watch for oncoming traffic
                - Do not overtake
                """)
            else:
                st.success(f"🟢 **STRAIGHT ROAD**\nConfidence: {confidence:.1f}%\n⏱️ {elapsed:.0f}ms")
                st.info("""
                ✅ **Driving Advice — Straight Road**
                - Normal driving conditions
                - Maintain safe following distance
                - Stay alert and focused
                """)
            
            # Confidence chart
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            
            bar_colors = ["#E24B4A", "#1D9E75"]
            bar_values = [probs[i].item() * 100 for i in range(2)]
            bars = ax.barh(CLASSES, bar_values, color=bar_colors, height=0.4)
            
            for bar, val in zip(bars, bar_values):
                ax.text(min(val + 1, 90), bar.get_y() + bar.get_height()/2, 
                        f"{val:.1f}%", va='center', fontsize=12, 
                        fontweight='bold', color='white')
            
            ax.set_xlim(0, 100)
            ax.set_xlabel("Confidence (%)", color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            plt.tight_layout()
            st.pyplot(fig)

with col2:
    st.markdown("### 📊 Model Performance")
    st.metric("Accuracy", "90.2%")
    st.metric("Sharp - Precision", "92%")
    st.metric("Sharp - Recall", "89%")
    st.metric("Straight - Precision", "89%")
    st.metric("Straight - Recall", "92%")

st.markdown("---")
st.markdown("""
**Model:** MobileNetV2 | **Framework:** PyTorch | **Deployment:** Streamlit Cloud
""")
