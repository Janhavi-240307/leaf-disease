"""Streamlit Web Application for Potato Leaf Disease Classification and OpenCV Inspection."""

import sys
from pathlib import Path

# Add project root to path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import CLASS_NAMES, DISPLAY_NAMES, MODEL_PATH, SAMPLES_DIR
from src.inference import PotatoDiseasePredictor

# Page Configuration
st.set_page_config(
    page_title="Plant Leaf Disease AI - Potato Care",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_predictor():
    """Initializes and caches the disease predictor."""
    return PotatoDiseasePredictor(model_path=MODEL_PATH)


def main():
    # Header
    st.title("🌿 Plant Leaf Disease Detection System")
    st.caption("Powered by **TensorFlow/Keras (MobileNetV2)**, **OpenCV**, and **PlantVillage Dataset**")

    predictor = get_predictor()
    model_ready = predictor.is_model_ready()

    # Sidebar Information
    with st.sidebar:
        st.header("⚙️ System Status")
        if model_ready:
            st.success("✅ Model Loaded (`models/potato_disease_model.keras`)")
        else:
            st.warning("⚠️ Model Not Trained Yet")
            st.info(
                "Run `python src/train.py` in your terminal to train and export the model."
            )

        st.markdown("---")
        st.subheader("🌾 Target Plant & Classes")
        st.markdown("""
        **Crop:** Potato (*Solanum tuberosum*)  
        **Supported Classes:**
        1. 🟤 `Early Blight` (*Alternaria solani*)
        2. 🔴 `Late Blight` (*Phytophthora infestans*)
        3. 🟢 `Healthy Leaf`
        """)

        st.markdown("---")
        st.subheader("🔬 OpenCV Pipeline")
        st.markdown("""
        - **HSV Color Masking:** Separates plant foliage from background noise.
        - **CLAHE Enhancement:** Intensifies subtle fungal lesion spots in LAB space.
        - **Contour Extraction:** Detects leaf perimeter and bounding box.
        """)

    # Top notification banner if model is uninitialized
    if not model_ready:
        st.warning(
            "⚠️ **Notice:** Trained model weights were not detected on disk. "
            "You can still inspect the **OpenCV Image Preprocessor** below! "
            "To enable deep learning classification, train the model first by running: `python src/train.py`."
        )

    # Input Mode Selection
    st.subheader("1. Provide a Leaf Image")
    input_source = st.radio(
        "Choose Input Method:",
        options=["Upload File", "Capture via Webcam", "Use Pre-loaded Sample"],
        horizontal=True,
    )

    image_bytes = None

    if input_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload a potato leaf photo (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()

    elif input_source == "Capture via Webcam":
        st.caption("Click 'Take Photo' below to capture a leaf with your camera.")
        camera_file = st.camera_input("Capture leaf")
        if camera_file is not None:
            image_bytes = camera_file.read()

    elif input_source == "Use Pre-loaded Sample":
        sample_files = list(SAMPLES_DIR.glob("**/*.jpg"))
        if sample_files:
            selected_sample = st.selectbox(
                "Select a sample image:",
                options=sample_files,
                format_func=lambda p: f"{p.parent.name} - {p.name}",
            )
            if selected_sample and selected_sample.exists():
                with open(selected_sample, "rb") as f:
                    image_bytes = f.read()
        else:
            st.info("No sample images found. Run `python src/dataset_setup.py --create-samples` to generate test samples.")

    # Processing & Results Section
    if image_bytes is not None:
        st.markdown("---")
        st.subheader("2. Diagnostic Analysis & OpenCV Preprocessing")

        with st.spinner("Processing image through OpenCV and CNN..."):
            result = predictor.predict(image_bytes)

        views = result["views"]

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("### 🔍 Computer Vision Inspection")
            view_tabs = st.tabs([
                "Original",
                "Segmented Leaf",
                "Lesion Enhanced",
                "Binary Mask",
                "Contour & Bounds",
            ])

            with view_tabs[0]:
                st.image(views["original_rgb"], caption="Original Input", use_container_width=True)
            with view_tabs[1]:
                st.image(views["segmented_rgb"], caption="OpenCV HSV Foliage Segmentation", use_container_width=True)
            with view_tabs[2]:
                st.image(views["enhanced_rgb"], caption="CLAHE Contrast Enhanced Lesions", use_container_width=True)
            with view_tabs[3]:
                st.image(views["mask"], caption="Binary Segmentation Mask", use_container_width=True)
            with view_tabs[4]:
                st.image(views["contour_rgb"], caption="Detected Leaf Contour & Bounding Box", use_container_width=True)

        with col2:
            st.markdown("### 🧪 Classification Diagnosis")

            if not result.get("model_ready", False):
                st.error("Model weights are missing. Deep learning diagnosis is unavailable until training is completed.")
            else:
                predicted_class = result["predicted_class"]
                display_name = result["display_name"]
                confidence = result["confidence"]
                probabilities = result["probabilities"]

                # Status Badge
                if "healthy" in predicted_class:
                    st.success(f"### Result: {display_name}")
                elif "Early_blight" in predicted_class:
                    st.warning(f"### Result: {display_name}")
                else:
                    st.error(f"### Result: {display_name}")

                st.metric(
                    label="Prediction Confidence",
                    value=f"{confidence * 100:.2f}%",
                )
                st.progress(min(float(confidence), 1.0))

                # Probability Breakdown Bar Chart
                st.markdown("#### Probability Distribution")
                chart_data = pd.DataFrame({
                    "Condition": [DISPLAY_NAMES.get(c, c) for c in probabilities.keys()],
                    "Probability": [prob * 100 for prob in probabilities.values()],
                }).set_index("Condition")

                st.bar_chart(chart_data)

        # Agronomic Guidance Card
        if result.get("model_ready", False) and "disease_info" in result:
            info = result["disease_info"]
            st.markdown("---")
            st.subheader(f"3. Agronomic Treatment & Prevention Plan ({info.get('common_name', 'General')})")

            t1, t2, t3, t4 = st.tabs([
                "📋 Symptoms & Pathogen",
                "🌿 Organic / Cultural Control",
                "🧪 Chemical Treatment",
                "🛡️ Prevention Protocol",
            ])

            with t1:
                st.write(f"**Pathogen:** {info.get('pathogen', 'N/A')}")
                st.write(f"**Severity Level:** {info.get('severity', 'N/A')}")
                st.markdown("**Key Symptoms:**")
                for s in info.get("symptoms", []):
                    st.markdown(f"- {s}")
                st.markdown("**Favorable Environmental Conditions:**")
                for c in info.get("favorable_conditions", []):
                    st.markdown(f"- {c}")

            with t2:
                st.markdown("**Recommended Organic & Cultural Actions:**")
                for o in info.get("organic_control", []):
                    st.markdown(f"- {o}")

            with t3:
                st.markdown("**Chemical Fungicide Recommendations:**")
                for ch in info.get("chemical_control", []):
                    st.markdown(f"- {ch}")

            with t4:
                st.markdown("**Long-term Prevention & Field Management:**")
                for p in info.get("prevention", []):
                    st.markdown(f"- {p}")


if __name__ == "__main__":
    main()

