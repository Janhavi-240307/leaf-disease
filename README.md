# 🌱 Plant Leaf Disease Classification System

An end-to-end Computer Vision and Deep Learning system for classifying Potato leaf diseases using **TensorFlow/Keras (MobileNetV2 Transfer Learning)**, **OpenCV**, and **Streamlit**.

---

## 📌 Project Overview

- **Crop Target:** Potato (*Solanum tuberosum*)
- **Target Classes:**
  1. `Potato___Early_blight` (*Alternaria solani*)
  2. `Potato___Late_blight` (*Phytophthora infestans*)
  3. `Potato___healthy`
- **Authentic Dataset:** PlantVillage Benchmark Dataset (**2,152 total verified images** across all 3 classes).
- **Core Technologies:**
  - **OpenCV (`cv2`):** HSV color masking for background suppression, CLAHE contrast enhancement for necrotic lesion textures, and contour/bounding box extraction.
  - **Deep Learning (TensorFlow/Keras):** Transfer Learning with **MobileNetV2** (ImageNet pretrained backbone) fine-tuned on potato pathology.
  - **Streamlit:** Presentation-ready web application supporting both file upload and live browser webcam capture, OpenCV diagnostic tabs, probability meters, and agronomic management advice.

---

## 🗂️ Project Directory Structure

```
leaf-disease/
│
├── .venv/                      # Python 3.11.9 virtual environment (configured)
├── app.py                      # Main presentation-ready Streamlit web application
├── train.py                    # Root model training & evaluation pipeline
├── predict.py                  # CLI & programmatic OpenCV + CNN inference utility
├── dataset_loader.py           # Authentic dataset downloader & OpenCV validator
├── requirements.txt            # Dependency specifications (TensorFlow, OpenCV, Streamlit, etc.)
├── README.md                   # Complete documentation and Windows execution guide
├── .gitignore                  # Git ignore rules for environments, caches, and raw weights
│
├── data/
│   ├── raw/                    # Genuine PlantVillage Potato class folders (2,152 images)
│   │   ├── Potato___Early_blight/   (1,000 images)
│   │   ├── Potato___Late_blight/    (1,000 images)
│   │   └── Potato___healthy/        (152 images)
│   └── samples/                # Curated sample images for instant app testing
│
├── src/
│   ├── config.py               # Centralized hyperparameters, directories, and classes
│   ├── disease_info.py         # Agronomic knowledge base (symptoms, remedies, prevention)
│   ├── preprocessor.py         # OpenCV image segmentation and enhancement pipeline
│   ├── model.py                # MobileNetV2 transfer learning & Custom CNN definitions
│   ├── evaluate.py             # Evaluation module (classification report & confusion matrix)
│   └── inference.py            # Unified OpenCV + Keras inference engine
│
└── models/
    ├── potato_disease_model.keras  # Saved trained model weights
    ├── class_indices.json          # Numerical index-to-class label mapping
    ├── metrics.json                # Recorded training and validation performance metrics
    ├── training_history.png        # Training accuracy and loss plots across epochs
    ├── confusion_matrix.png        # Validation set confusion matrix heatmap
    └── classification_report.txt   # Precision, recall, and F1-score report
```

---

## 🚀 Quick Start Guide (Windows)

All dependencies and model weights are fully installed and configured in your `.venv` environment.

### 1. Activate the Virtual Environment

In your PowerShell terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

---

### 2. Launch the Streamlit Web Application

The primary launch command is:

```powershell
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

#### What you can do in the application:
1. **📁 Upload Image File:** Drag and drop any potato leaf photo (JPG/PNG).
2. **📷 Live Webcam Capture:** Grant browser camera permissions, position a leaf in frame, and click **Take photo** for an instant real-time diagnostic scan.
3. **🧪 Select Dataset Sample:** Choose from pre-loaded verified leaf samples to immediately demonstrate functionality.
4. **🔍 Computer Vision Inspection:** Switch between side-by-side tabs to inspect the **Original Image**, **HSV Foliage Mask**, **CLAHE Lesion Enhancement**, **Binary Mask**, and **Contour Bounding Box**.
5. **📊 Model Performance Dashboard:** Switch to the second tab to view live training curves, confusion matrix heatmap, and verified dataset distributions.

---

### 3. Test Predictions via Command Line

You can run automated inference on any leaf image using `predict.py`:

```powershell
# Run self-test on a sample image:
python predict.py --test

# Or test on a specific leaf file:
python predict.py --image data/samples/Potato___Early_blight/sample_1.jpg --save-vis models/output_vis.jpg
```

---

### 4. (Optional) Re-train the Model

The model is already trained and saved at `models/potato_disease_model.keras`. If you wish to re-train or experiment with hyperparameters:

```powershell
# Train for 6 epochs with batch size 32:
python train.py --epochs 6 --batch-size 32
```

---

### 5. (Optional) Verify Dataset Integrity

To re-scan and validate all dataset images with OpenCV:

```powershell
python dataset_loader.py
```

---

## 📊 Real Training & Evaluation Metrics

- **Backbone Architecture:** MobileNetV2 (ImageNet Pre-trained) + Custom Dense Classifier Head
- **Dataset Size:** 2,152 authentic PlantVillage images (80% Train, 20% Validation)
- **Validation Accuracy:** **75.35%** (Trained for 6 epochs on CPU)
- **Key Metrics:**
  - *Early Blight Precision:* 100.00%
  - *Late Blight Precision:* 93.58%
  - *Healthy Recall:* 100.00%
- Artifacts generated:
  - Model weights: `models/potato_disease_model.keras`
  - Training curves: `models/training_history.png`
  - Confusion matrix heatmap: `models/confusion_matrix.png`

---

## 🛠️ Troubleshooting

| Issue | Resolution |
| :--- | :--- |
| **PowerShell script execution disabled** | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell. |
| **Webcam permission denied in browser** | Click the camera icon in your browser address bar and choose "Allow access to camera". |
| **Port 8501 in use** | Run `streamlit run app.py --server.port 8502` to start on an alternative port. |
