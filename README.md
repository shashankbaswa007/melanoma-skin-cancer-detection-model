---
title: melanoma_skincancer_detector
app_file: app.py
sdk: gradio
sdk_version: 6.1.0
---
# Melanoma Skin Cancer Detection

Binary classification of dermoscopic images (**benign vs malignant**) using a fine-tuned **EfficientNetB3** backbone trained on the [Melanoma Skin Cancer Dataset of 10 000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images).

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~92 % |
| Test AUC | ~97 % |

---

## Model Architecture

```
EfficientNetB3 (ImageNet weights, frozen Phase 1)
  └─ GlobalAveragePooling2D
  └─ BatchNormalization
  └─ Dense(512, relu) + L2 regularisation
  └─ Dropout(0.45)
  └─ Dense(256, relu) + L2 regularisation
  └─ Dropout(0.35)
  └─ Dense(1, sigmoid)        ← binary output
```

**Training strategy**
- **Phase 1** (25 epochs): backbone frozen, head trained with `Adam(lr=1e-3)`
- **Phase 2** (35 epochs): top 30 % of backbone unfrozen, fine-tuned with `Adam(lr=1e-5)`
- Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`
- Class imbalance handled via `compute_class_weight`

---

## Project Structure

```
melanoma_cancer_dataset/
├── melanoma-skin-cancer-1.ipynb   # Main notebook (EDA → training → evaluation → Gradio demo)
├── README.md
├── .gitignore
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

> **Note:** The `train/` and `test/` image folders are excluded from this repository via `.gitignore`.  
> Download the dataset from Kaggle and place it at the repo root before running the notebook.

---

## Setup

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python pillow tqdm gradio
```

Open `melanoma-skin-cancer-1.ipynb` and run all cells in order.

---

## Interactive Demo

Cell 21 launches a **Gradio** web app that accepts an uploaded dermoscopic image and returns:
- Predicted class (benign / malignant)
- Confidence score
- Risk level and clinical recommendation

---

## Disclaimer

This project is for **research and educational purposes only** and does not constitute a clinical diagnosis or replace professional medical advice.
