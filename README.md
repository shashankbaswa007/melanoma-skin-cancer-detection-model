# 🔬 Melanoma Skin Cancer Detection

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gradio-6.x-F97316?style=for-the-badge&logo=gradio&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
</p>

<p align="center">
  <b>Binary classification of dermoscopic skin lesion images (benign vs malignant) using a fine-tuned EfficientNetB3 deep learning model.</b>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/shashankbaswa007/melanoma-skincancer-detector">🚀 Live Demo</a> &nbsp;|&nbsp;
  <a href="https://www.kaggle.com/code/shashankbaswa/melanoma-skin-cancer-1/edit">📓 Kaggle Notebook</a> &nbsp;|&nbsp;
  <a href="https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images">📦 Dataset</a>
</p>

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~92 % |
| Test AUC | ~97 % |
| Test ROC-AUC | ~97 % |

---

## 🧠 Model Architecture

Transfer learning with **EfficientNetB3** (pretrained on ImageNet), fine-tuned in two phases:

```
Input (224 × 224 × 3)
  └─ EfficientNetB3 backbone (ImageNet weights)
       └─ GlobalAveragePooling2D
       └─ BatchNormalization
       └─ Dense(512, ReLU) + L2 regularisation
       └─ Dropout(0.45)
       └─ Dense(256, ReLU) + L2 regularisation
       └─ Dropout(0.35)
       └─ Dense(1, Sigmoid)  →  P(malignant)
```

### Training Strategy

| Phase | Epochs | Backbone | Learning Rate |
|---|---|---|---|
| Phase 1 — Head training | 25 | Frozen | 1e-3 |
| Phase 2 — Fine-tuning | 35 | Top 30 % unfrozen | 1e-5 |

- **Callbacks:** `ModelCheckpoint`, `EarlyStopping` (patience=10), `ReduceLROnPlateau`
- **Class imbalance:** handled via `sklearn.utils.class_weight.compute_class_weight`
- **Augmentation:** rotation, shifts, zoom, flips, brightness jitter

---

## 🗂️ Project Structure

```
melanoma-skin-cancer-detection-model/
├── melanoma-skin-cancer-1.ipynb   # Full pipeline: EDA → train → evaluate → Gradio demo
├── app.py                         # Root-level Gradio app (local use)
├── requirements.txt
├── push_model_to_hub.py           # Upload trained model to HF Hub
├── .gitignore
├── space/
│   ├── app.py                     # HF Spaces deployment
│   └── requirements.txt
├── train/                         # ← excluded via .gitignore
│   ├── benign/
│   └── malignant/
└── test/                          # ← excluded via .gitignore
    ├── benign/
    └── malignant/
```

---

## ⚙️ Local Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/shashankbaswa007/melanoma-skin-cancer-detection-model.git
cd melanoma-skin-cancer-detection-model
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python pillow tqdm gradio huggingface_hub
```

### 2. Download the dataset

Download the [Melanoma Skin Cancer Dataset of 10 000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) from Kaggle and place `train/` and `test/` folders at the repo root.

### 3. Run the notebook

Open `melanoma-skin-cancer-1.ipynb` in VS Code or JupyterLab and **Run All**.  
The notebook auto-detects the local dataset path — no manual path editing needed.

### 4. Run the Gradio app locally

```bash
python app.py
```

---

## 🚀 Live Demo

The app is deployed on **Hugging Face Spaces**:

👉 **[https://huggingface.co/spaces/shashankbaswa007/melanoma-skincancer-detector](https://huggingface.co/spaces/shashankbaswa007/melanoma-skincancer-detector)**

Upload a dermoscopic image to receive:
- Predicted class: **Benign** or **Malignant**
- Confidence score
- Risk level (🟢 Low / 🟡 Low–Moderate / 🟠 Moderate / 🔴 High)
- Clinical recommendation

---

## 📓 Kaggle Notebook

The full training pipeline is also available on Kaggle with free GPU access:

👉 **[https://www.kaggle.com/code/shashankbaswa/melanoma-skin-cancer-1](https://www.kaggle.com/code/shashankbaswa/melanoma-skin-cancer-1/edit)**

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow 2.20 / Keras |
| Backbone | EfficientNetB3 (ImageNet) |
| Data Augmentation | Keras `ImageDataGenerator` |
| Evaluation | scikit-learn, matplotlib, seaborn |
| Web App | Gradio 6.x |
| Deployment | Hugging Face Spaces |
| Model Storage | Hugging Face Hub |

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only**.  
It does **not** constitute a clinical diagnosis and should **not** replace consultation with a qualified dermatologist.  
All skin lesions of concern must be evaluated by a licensed medical professional.
