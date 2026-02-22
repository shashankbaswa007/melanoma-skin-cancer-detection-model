"""
push_model_to_hub.py
────────────────────
Upload the trained Keras model to a Hugging Face Hub model repository
so that app.py can download it at runtime on HF Spaces.

Usage
-----
  python push_model_to_hub.py --model best_melanoma_model.keras

Prerequisites
-------------
  pip install huggingface_hub
  huggingface-cli login        # paste your HF write token when prompted
"""

import argparse
from huggingface_hub import HfApi, create_repo

HF_REPO_ID = 'shashankbaswa007/melanoma-efficientnetb3'   # ← your HF username/repo-name


def push(model_path: str):
    api = HfApi()

    # Create the repo if it doesn't exist (model card type, public)
    create_repo(repo_id=HF_REPO_ID, repo_type='model', exist_ok=True)
    print(f"Repository ready: https://huggingface.co/{HF_REPO_ID}")

    print(f"Uploading {model_path} …")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo='best_melanoma_model.keras',
        repo_id=HF_REPO_ID,
        repo_type='model',
    )
    print("Upload complete.")
    print(f"Model URL: https://huggingface.co/{HF_REPO_ID}/resolve/main/best_melanoma_model.keras")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best_melanoma_model.keras', help='Path to .keras model file')
    args = parser.parse_args()
    push(args.model)
