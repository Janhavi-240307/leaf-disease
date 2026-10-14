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
                image_data = uploaded_file.getvalue()

        elif input_choice == "📷 Live Webcam Capture":
            st.info("💡 **Camera Instructions:** Position a potato leaf in front of your camera. Ensure even lighting and steady focus.")
            camera_image = st.camera_input("Take a photo of the plant leaf")
            if camera_image is not None:
                image_data = camera_image.getvalue()

        elif input_choice == "🧪 Select Dataset Sample":
            # Search samples directory (available both locally and on Streamlit Cloud)
            candidates = sorted(
                list(SAMPLES_DIR.glob("**/*.jpg")) +
                list(SAMPLES_DIR.glob("**/*.JPG")) +
                list(SAMPLES_DIR.glob("**/*.png"))
            )
            if not candidates and RAW_DATA_DIR.exists():
                candidates = list(RAW_DATA_DIR.glob("**/*.jpg"))[:20]

            if candidates:
                sample_path = st.selectbox(
                    "Choose an authentic sample image:",
                    options=candidates,
                    format_func=lambda p: f"{p.parent.name.replace('Potato___', '')} - {p.name}",
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
                    if image_data is not None and isinstance(image_data, (bytes, bytearray)):
                        st.image(image_data, caption="Original Input Image", use_container_width=True)
                    elif views.get("original_rgb") is not None:
                        st.image(views["original_rgb"], caption="Original Input Image", use_container_width=True)

                with cv_tabs[1]:
                    if views.get("segmented_rgb") is not None:
                        st.image(views["segmented_rgb"], caption="Background-Suppressed Foliage (HSV Mask)", use_container_width=True)

                with cv_tabs[2]:
                    if views.get("enhanced_rgb") is not None:
                        st.image(views["enhanced_rgb"], caption="CLAHE Contrast-Enhanced Lesion Spots", use_container_width=True)

                with cv_tabs[3]:
                    if views.get("mask") is not None:
                        st.image(views["mask"], caption="Binary Foliage Mask", use_container_width=True)

                with cv_tabs[4]:
                    if views.get("contour_rgb") is not None:
                        st.image(views["contour_rgb"], caption="Largest Contour & Bounding Box", use_container_width=True)

            # Right Column: Disease Diagnosis
            with col2:
                st.subheader("🧪 Deep Learning Diagnosis")

                if not result.get("success", False):
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

                    # Confidence Metric & Certainty Level
                    col_m1, col_m2 = st.columns([1, 1])
                    with col_m1:
                        st.metric("Model Confidence Score", f"{conf * 100:.2f}%")
                    with col_m2:
                        if conf >= 0.75:
                            st.caption("🟢 **High Diagnostic Certainty**")
                        elif conf >= 0.50:
                            st.caption("🟡 **Moderate Diagnostic Certainty**")
                        else:
                            st.caption("🟠 **Borderline / Mixed Features** (Split probability across classes)")

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
                st.markdown("##### 📁 Verified Dataset Distribution & Split Partition")
                support = metrics.get("class_support", {})
                counts = metrics["class_counts"]
                rows = []
                for k, total_cnt in counts.items():
                    val_cnt = support.get(k, int(round(total_cnt * 0.2)))
                    train_cnt = total_cnt - val_cnt
                    rows.append({
                        "Condition": DISPLAY_NAMES.get(k, k),
                        "Total Images": total_cnt,
                        "Training Split (80%)": train_cnt,
                        "Validation Split (20%)": val_cnt,
                    })
                # Add total summary row
                total_all = sum(r["Total Images"] for r in rows)
                train_all = sum(r["Training Split (80%)"] for r in rows)
                val_all = sum(r["Validation Split (20%)"] for r in rows)
                rows.append({
                    "Condition": "TOTAL",
                    "Total Images": total_all,
                    "Training Split (80%)": train_all,
                    "Validation Split (20%)": val_all,
                })
                counts_df = pd.DataFrame(rows)
                st.dataframe(counts_df, hide_index=True, use_container_width=True)

        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.markdown("##### 📈 Training & Validation Loss/Accuracy Curves")
            if HISTORY_PLOT_PATH.exists():
                try:
                    from PIL import Image
                    hist_img = Image.open(HISTORY_PLOT_PATH)
                    st.image(hist_img, caption="Training Progress Across Epochs (MobileNetV2)", use_container_width=True)
                except Exception:
                    st.image(str(HISTORY_PLOT_PATH), caption="Training Progress Across Epochs (MobileNetV2)", use_container_width=True)
            else:
                st.info("Training history plot will be displayed here once training is completed.")

        with col_img2:
            st.markdown("##### 🎯 Validation Confusion Matrix")
            cm_data = metrics.get("confusion_matrix") if metrics else None
            classes_list = metrics.get("classes", CLASS_NAMES) if metrics else CLASS_NAMES
            if cm_data is not None:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    fig, ax = plt.subplots(figsize=(6, 5))
                    cm_array = np.array(cm_data)
                    labels = [name.replace("Potato___", "").replace("_", " ") for name in classes_list]
                    sns.heatmap(
                        cm_array,
                        annot=True,
                        fmt="d",
                        cmap="Greens",
                        xticklabels=labels,
                        yticklabels=labels,
                        cbar=True,
                        ax=ax,
                    )
                    val_n = metrics.get("val_samples", 430)
                    ax.set_title(f"Confusion Matrix (N={val_n} Validation Samples)", fontsize=12, pad=10)
                    ax.set_xlabel("Predicted Label", fontsize=10)
                    ax.set_ylabel("True Label", fontsize=10)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception:
                    if CONFUSION_MATRIX_PATH.exists():
                        try:
                            from PIL import Image
                            cm_img = Image.open(CONFUSION_MATRIX_PATH)
                            st.image(cm_img, caption="Confusion Matrix on Validation Data", use_container_width=True)
                        except Exception:
                            st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix on Validation Data", use_container_width=True)
            elif CONFUSION_MATRIX_PATH.exists():
                try:
                    from PIL import Image
                    cm_img = Image.open(CONFUSION_MATRIX_PATH)
                    st.image(cm_img, caption="Confusion Matrix on Validation Data", use_container_width=True)
                except Exception:
                    st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix on Validation Data", use_container_width=True)
            else:
                st.info("Confusion matrix will be displayed here once evaluation is completed.")

        # Show Classification Report if saved
        report_file = MODELS_DIR / "classification_report.txt"
        if report_file.exists():
            st.markdown("##### 📑 Detailed Classification Report (All 3 Classes Evaluated)")
            with open(report_file, "r") as f:
                st.code(f.read(), language="text")


if __name__ == "__main__":
    main()

