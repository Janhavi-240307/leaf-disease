"""Root-level College-Presentation Ready Streamlit Web Application.
Features:
- Image Upload and Live Browser Webcam Capture
- OpenCV Computer Vision preprocessing pipeline (HSV Segmentation, CLAHE lesion highlighting, Contours)
- Real-time CNN Classification with confidence scoring and probability distributions
- Comprehensive agronomic disease guidance (symptoms, organic controls, fungicides, prevention)
- Interactive Model Performance Dashboard displaying training metrics and confusion matrix
"""

import sys
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np

# Ensure root is in system path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    MODEL_PATH,
    MODELS_DIR,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    HISTORY_PLOT_PATH,
    RAW_DATA_DIR,
    SAMPLES_DIR,
)
from predict import DiseaseClassifier

# Page Configuration
st.set_page_config(
    page_title="AgriVision AI - Plant Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #424242;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2e7d32;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 6px 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    """Initializes and caches the trained disease classifier."""
    return DiseaseClassifier(model_path=MODEL_PATH)


def load_metrics_data():
    """Loads saved training metrics metadata if available."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    classifier = load_classifier()
    model_ready = classifier.is_ready()
    metrics = load_metrics_data()

    # Application Header
    st.markdown('<div class="main-title">🌱 AgriVision: Plant Leaf Disease Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Intelligent Agricultural Diagnostics using <b>Convolutional Neural Networks (MobileNetV2)</b>, <b>OpenCV</b>, and <b>PlantVillage Dataset</b></div>',
        unsafe_allow_html=True,
    )

    # Sidebar Navigation & System Telemetry
    with st.sidebar:
        st.header("🔬 Project Telemetry")
        if model_ready:
            st.success("✅ **CNN Model Loaded** (`potato_disease_model.keras`)")
        else:
            st.warning("⚠️ **Model Weights Missing**")
            st.info("Run `python train.py` to train and export weights.")

        if metrics and "validation_accuracy" in metrics:
            val_acc = metrics["validation_accuracy"] * 100
            st.metric("Model Validation Accuracy", f"{val_acc:.2f}%")
            if "epochs_trained" in metrics:
                st.caption(f"Trained for {metrics['epochs_trained']} epochs.")

        st.markdown("---")
        st.subheader("🌾 Target Crop & Pathogens")
        st.markdown("""
        **Host:** Potato (*Solanum tuberosum*)  
        - 🟤 **Early Blight** (*Alternaria solani*)  
        - 🔴 **Late Blight** (*Phytophthora infestans*)  
        - 🟢 **Healthy Foliage**
        """)

        st.markdown("---")
        st.subheader("🧪 OpenCV Processing Pipeline")
        st.markdown("""
        1. **HSV Color Masking:** Segments leaf body from noisy background.
        2. **CLAHE Equalization:** Intensifies subtle necrotic spot contrast.
        3. **Contour Extraction:** Detects perimeter and bounding box.
        """)

        st.markdown("---")
        st.caption("Developed for College Capstone / Engineering Demonstration.")

    # Main Tabs: Diagnostics & Evaluation Dashboard
    main_tab1, main_tab2 = st.tabs(["🩺 Leaf Disease Diagnosis", "📊 Model Performance & Evaluation"])

    with main_tab1:
        if not model_ready:
            st.warning(
                "⚠️ **Note:** The CNN model weights are not yet generated. "
                "OpenCV preprocessing and sample inspection are available below. "
                "Execute `python train.py` in your terminal to train and activate automated classification."
            )

        st.markdown("#### 1. Input Leaf Image")
        input_choice = st.radio(
            "Select Input Source:",
            ["📁 Upload Image File", "📷 Live Webcam Capture", "🧪 Select Dataset Sample"],
            horizontal=True,
        )

        image_data = None

        if input_choice == "📁 Upload Image File":
            uploaded_file = st.file_uploader(
                "Choose a leaf photo (JPG, JPEG, PNG)...",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear picture of a potato plant leaf against a neutral or natural background.",
            )
            if uploaded_file is not None:
                image_data = uploaded_file.read()

        elif input_choice == "📷 Live Webcam Capture":
            st.info("💡 **Camera Instructions:** Position a potato leaf in front of your camera. Ensure even lighting and steady focus.")
            camera_image = st.camera_input("Take a photo of the plant leaf")
            if camera_image is not None:
                image_data = camera_image.read()

        elif input_choice == "🧪 Select Dataset Sample":
            # Search dataset folders for candidate samples
            candidates = (
                list(RAW_DATA_DIR.glob("**/*.jpg")) +
                list(RAW_DATA_DIR.glob("**/*.png")) +
                list(SAMPLES_DIR.glob("**/*.jpg"))
            )
            if candidates:
                sample_path = st.selectbox(
                    "Choose an authentic sample image:",
                    options=candidates,
                    format_func=lambda p: f"{p.parent.name} - {p.name}",
                )
                if sample_path:
                    with open(sample_path, "rb") as f:
                        image_data = f.read()
            else:
                st.info("No sample images discovered. Download dataset or run `python dataset_loader.py`.")

        # Diagnosis and Preprocessing Results
        if image_data is not None:
            st.markdown("---")
            st.markdown("#### 2. Computer Vision Inspection & Deep Learning Diagnosis")

            # Check if this is a newly loaded image or cached
            if "last_image_data" not in st.session_state or st.session_state["last_image_data"] != image_data:
                st.session_state["last_image_data"] = image_data
                st.session_state["result"] = None

            c_btn1, c_btn2 = st.columns([1, 2])
            with c_btn1:
                predict_clicked = st.button("🔬 Predict Disease & Process Leaf", type="primary", use_container_width=True)

            if predict_clicked or st.session_state.get("result") is None:
                with st.spinner("Processing image through OpenCV pipeline and CNN..."):
                    st.session_state["result"] = classifier.predict(image_data)

            result = st.session_state["result"]
            views = result.get("views", {})

            col1, col2 = st.columns([1, 1], gap="large")

            # Left Column: OpenCV Pipeline Inspection
            with col1:
                st.subheader("🔍 OpenCV Image Analysis")
                cv_tabs = st.tabs([
                    "Original Input",
                    "HSV Segmented Leaf",
                    "CLAHE Enhanced Lesions",
                    "Binary Foliage Mask",
                    "Contour & Bounds",
                ])

                with cv_tabs[0]:
                    st.image(views.get("original_rgb"), caption="Input Image", use_container_width=True)
                with cv_tabs[1]:
                    st.image(views.get("segmented_rgb"), caption="Background-Suppressed Foliage (HSV Mask)", use_container_width=True)
                with cv_tabs[2]:
                    st.image(views.get("enhanced_rgb"), caption="CLAHE Contrast-Enhanced Lesion Spots", use_container_width=True)
                with cv_tabs[3]:
                    st.image(views.get("mask"), caption="Binary Foliage Mask", use_container_width=True)
                with cv_tabs[4]:
                    st.image(views.get("contour_rgb"), caption="Largest Contour & Bounding Box", use_container_width=True)

            # Right Column: Disease Diagnosis
            with col2:
                st.subheader("🧪 Deep Learning Diagnosis")

                if not result["success"]:
                    st.error(result.get("message", "Model prediction failed."))
                else:
                    pred_class = result["predicted_class"]
                    disp_name = result["display_name"]
                    conf = result["confidence"]
                    probs = result["probabilities"]

                    # Status Banner
                    if "healthy" in pred_class:
                        st.success(f"### Diagnosis: {disp_name}")
                    elif "Early_blight" in pred_class:
                        st.warning(f"### Diagnosis: {disp_name}")
                    else:
                        st.error(f"### Diagnosis: {disp_name}")

                    # Confidence Metric
                    st.metric("Model Confidence Score", f"{conf * 100:.2f}%")
                    st.progress(min(float(conf), 1.0))

                    # Probability Breakdown
                    st.markdown("##### Class Probability Distribution")
                    chart_df = pd.DataFrame({
                        "Class": [DISPLAY_NAMES.get(k, k) for k in probs.keys()],
                        "Probability (%)": [v * 100 for v in probs.values()],
                    }).set_index("Class")
                    st.bar_chart(chart_df)

            # Bottom Section: Agronomic Action Plan
            if result["success"] and "disease_info" in result and result["disease_info"]:
                info = result["disease_info"]
                st.markdown("---")
                st.markdown(f"#### 3. Agronomic Management & Treatment Plan: *{info.get('common_name', 'General')}*")
                st.info("ℹ️ **Disclaimer:** This diagnosis is generated by an automated computer vision model for educational and decision-support purposes. Confirm critical findings with a local agronomist or extension officer.")

                r_col1, r_col2 = st.columns(2)

                with r_col1:
                    st.markdown("##### 📋 Symptoms & Environmental Factors")
                    st.write(f"**Pathogen:** `{info.get('pathogen', 'N/A')}`")
                    st.write(f"**Severity Level:** `{info.get('severity', 'N/A')}`")
                    st.markdown("**Diagnostic Symptoms:**")
                    for s in info.get("symptoms", []):
                        st.markdown(f"- {s}")
                    st.markdown("**Favorable Climate Triggers:**")
                    for c in info.get("favorable_conditions", []):
                        st.markdown(f"- {c}")

                with r_col2:
                    st.markdown("##### 🛡️ Remediation & Control Protocols")
                    st.markdown("**Organic / Biological Actions:**")
                    for o in info.get("organic_control", []):
                        st.markdown(f"- {o}")
                    st.markdown("**Chemical Fungicide Treatments:**")
                    for ch in info.get("chemical_control", []):
                        st.markdown(f"- {ch}")
                    st.markdown("**Preventive Cultural Practices:**")
                    for p in info.get("prevention", []):
                        st.markdown(f"- {p}")

    # Tab 2: Model Performance Dashboard
    with main_tab2:
        st.subheader("📊 Model Performance & Training Analytics")

        if metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Validation Accuracy", f"{metrics.get('validation_accuracy', 0) * 100:.2f}%")
            m2.metric("Validation Loss", f"{metrics.get('final_val_loss', 0):.4f}")
            m3.metric("Training Accuracy", f"{metrics.get('final_train_accuracy', 0) * 100:.2f}%")
            m4.metric("Epochs Completed", metrics.get("epochs_trained", "N/A"))

            if "class_counts" in metrics:
                st.markdown("##### Dataset Distribution Verified by OpenCV")
                counts_df = pd.DataFrame([
                    {"Condition": DISPLAY_NAMES.get(k, k), "Verified Images": v}
                    for k, v in metrics["class_counts"].items()
                ])
                st.dataframe(counts_df, hide_index=True, use_container_width=True)

        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.markdown("##### 📈 Training & Validation Loss/Accuracy Curves")
            if HISTORY_PLOT_PATH.exists():
                st.image(str(HISTORY_PLOT_PATH), caption="Training Progress Across Epochs", use_container_width=True)
            else:
                st.info("Training history plot will be displayed here once training is completed.")

        with col_img2:
            st.markdown("##### 🎯 Validation Confusion Matrix")
            if CONFUSION_MATRIX_PATH.exists():
                st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix on Validation Data", use_container_width=True)
            else:
                st.info("Confusion matrix will be displayed here once evaluation is completed.")

        # Show Classification Report if saved
        report_file = MODELS_DIR / "classification_report.txt"
        if report_file.exists():
            st.markdown("##### 📑 Detailed Classification Report")
            with open(report_file, "r") as f:
                st.code(f.read(), language="text")


if __name__ == "__main__":
    main()

