"""
Melanoma Skin Cancer Detection – Gradio app for HF Spaces
"""

import os
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from huggingface_hub import hf_hub_download

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
CLASS_NAMES = ['benign', 'malignant']

# ── Load model ─────────────────────────────────────────────────────────────────
def _load_model():
    local_path = 'best_melanoma_model.keras'
    if os.path.exists(local_path):
        print(f"Loading model from: {local_path}")
        return tf.keras.models.load_model(local_path)
    hf_repo = os.environ.get('HF_MODEL_REPO', 'shashankbaswa007/melanoma-efficientnetb3')
    print(f"Downloading model from HF Hub: {hf_repo}")
    model_path = hf_hub_download(repo_id=hf_repo, filename='best_melanoma_model.keras')
    return tf.keras.models.load_model(model_path)

model = _load_model()
print("Model loaded.")

# ── Inference helpers ──────────────────────────────────────────────────────────
def predict_image(pil_image, threshold=0.5):
    img = pil_image.resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = eff_preprocess(np.expand_dims(arr, axis=0))
    prob_malignant = float(model.predict(arr, verbose=0)[0][0])
    if prob_malignant >= threshold:
        return CLASS_NAMES[1], prob_malignant
    return CLASS_NAMES[0], 1.0 - prob_malignant


def _risk_level(pred_cls, conf):
    if pred_cls == 'malignant':
        if conf >= 0.80:
            return '🔴  HIGH RISK', 'Immediate dermatologist consultation is **strongly recommended**.'
        elif conf >= 0.60:
            return '🟠  MODERATE RISK', 'A dermatologist consultation is **recommended** at your earliest convenience.'
        return '🟡  LOW–MODERATE RISK', 'Consider **monitoring** this lesion and consulting a dermatologist.'
    return '🟢  LOW RISK', 'The lesion appears **benign**. Continue routine skin self-examination.'


def melanoma_inference(pil_image):
    if pil_image is None:
        return {c: 0.0 for c in CLASS_NAMES}, '*Please upload an image to proceed.*'

    pred_cls, conf = predict_image(pil_image)
    level, rec     = _risk_level(pred_cls, conf)

    p_mal = conf if pred_cls == 'malignant' else round(1.0 - conf, 4)
    confidence_dict = {
        'malignant': round(p_mal, 4),
        'benign':    round(1.0 - p_mal, 4),
    }
    report = f"""
### Result: **{pred_cls.upper()}**

| Field | Detail |
|---|---|
| Predicted class | `{pred_cls}` |
| Confidence | `{conf:.2%}` |
| Risk level | {level} |
| Recommendation | {rec} |

---
> **⚕️ Medical Disclaimer:** This tool is for **research and educational purposes only**
> and does **not** constitute a clinical diagnosis.
> All lesions of concern should be evaluated by a qualified dermatologist.
"""
    return confidence_dict, report


# ── Gradio layout ──────────────────────────────────────────────────────────────
_THEME = gr.themes.Soft(
    primary_hue='blue',
    secondary_hue='slate',
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'sans-serif'],
)

with gr.Blocks(title='Melanoma Detection') as demo:
    gr.Markdown("""
# 🔬 Melanoma Skin Cancer Detection
**AI-powered dermoscopic image analysis** using **EfficientNetB3** fine-tuned on the
Melanoma Skin Cancer dataset (10 000 images). Upload a skin lesion image to begin.
""")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=300):
            img_input = gr.Image(type='pil', label='Dermoscopic Image', height=300,
                                 sources=['upload', 'clipboard'])
            with gr.Row():
                analyse_btn = gr.Button('🔍  Analyse', variant='primary', size='lg')
                gr.ClearButton(components=[img_input], value='🗑  Clear', size='lg')

        with gr.Column(scale=1, min_width=300):
            label_out  = gr.Label(num_top_classes=2, label='Class Probabilities')
            report_out = gr.Markdown(
                value='*Upload an image and click **Analyse** to see the assessment.*')

    analyse_btn.click(fn=melanoma_inference, inputs=img_input,
                      outputs=[label_out, report_out], api_name='predict')
    img_input.upload(fn=melanoma_inference, inputs=img_input,
                     outputs=[label_out, report_out])

    gr.Markdown("""
---
**Model:** EfficientNetB3 &nbsp;|&nbsp; **Input:** 224×224 RGB &nbsp;|&nbsp;
**Classes:** Benign / Malignant &nbsp;|&nbsp; **Framework:** TensorFlow / Keras
""")

if __name__ == '__main__':
    demo.launch(theme=_THEME)
