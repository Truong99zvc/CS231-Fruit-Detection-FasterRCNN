import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
import numpy as np

# Project class names (same order as in training)
CLASS_NAMES = [
    "apple", "tangerine", "pear", "watermelon", "durian",
    "lemon", "grape", "pineapple", "dragon fruit", "korean melon", "cantaloupe"
]

MODEL_CHOICES = {
    "Train from Scratch (Scratch)": {
        "drive_id": "1sEtlFPUAjM2UgbUL3imPw6tdHGrgq9AW",
        "file": "model_scratch.pth"
    },
    "Backbone Pretrained (Backbone)": {
        "drive_id": "1_EX4m02uwZn7SQhNMLPUFqsOxStV1ZZI",
        "file": "model_backbone_pretrained.pth"
    },
    "Fine-tuned Full Model (FineTune)": {
        "drive_id": "1K2g9D4mpMKO_RICzXzTWCRAPbSNxi-rg",
        "file": "model_finetune_full.pth"
    }
}


def download_from_gdrive(drive_id: str, dest_path: str) -> bool:
    """Attempt to download a file from Google Drive using gdown.
    Returns True on success, False otherwise."""
    try:
        import gdown
    except Exception:
        return False

    url = f"https://drive.google.com/uc?id={drive_id}"
    try:
        gdown.download(url, dest_path, quiet=False)
        return os.path.exists(dest_path)
    except Exception:
        return False


def build_model(num_classes: int):
    # Build a Faster R-CNN model and replace the head
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@st.cache_resource
def load_model_weights(model_path: str, device: str = 'cpu'):
    # Build model and load state_dict
    num_model_classes = len(CLASS_NAMES) + 1  # include background
    model = build_model(num_model_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def draw_boxes_on_pil(img: Image.Image, boxes, labels, scores, label_map, conf_thresh=0.5):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for box, lbl, sc in zip(boxes, labels, scores):
        if sc < conf_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        text = f"{label_map.get(int(lbl), str(lbl))}: {sc:.2f}"
        text_pos = (x1, y1 - 16 if y1 - 16 > 0 else y1 + 4)
        text_bbox = draw.textbbox(text_pos, text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text(text_pos, text, fill="white", font=font)
    return img


def main():
    st.title("Fruit Detection — Faster R-CNN Demo")
    st.write("Simple demo app to load one of three trained models and run inference on uploaded images.")

    model_choice = st.selectbox("Choose model:", list(MODEL_CHOICES.keys()))
    model_info = MODEL_CHOICES[model_choice]
    models_dir = Path("./demo_streamlit/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / model_info['file']

    if not model_path.exists():
        st.info(f"Model file not found locally: {model_path}. Attempting to download from Google Drive.")
        with st.spinner("Downloading model (requires `gdown` Python package)..."):
            ok = download_from_gdrive(model_info['drive_id'], str(model_path))
        if not ok:
            st.error("Automatic download failed. Please install `gdown` (pip install gdown) or download manually from the link provided in the README.")
    else:
        st.success(f"Model ready: {model_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write(f"Running on device: {device}")

    if model_path.exists():
        if st.button("Load model"):
            with st.spinner("Loading model into memory..."):
                model = load_model_weights(str(model_path), device=device)
            st.success("Model loaded.")
            st.session_state['model'] = model

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.5)

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Input image", use_column_width=True)

        if 'model' in st.session_state:
            model = st.session_state['model']
            img_tensor = F.to_tensor(img).to(device)
            with st.spinner("Running inference..."):
                preds = model([img_tensor])[0]

            boxes = preds['boxes'].cpu().numpy()
            scores = preds['scores'].cpu().numpy()
            labels = preds['labels'].cpu().numpy()

            # map labels (1-11) to names
            label_map = {i+1: name for i, name in enumerate(CLASS_NAMES)}
            img_out = img.copy()
            img_out = draw_boxes_on_pil(img_out, boxes, labels, scores, label_map, conf_thresh=conf_thresh)
            st.image(img_out, caption="Predictions", use_column_width=True)
        else:
            st.warning("Model is not loaded. Click 'Load model' first or download the model file manually.")


if __name__ == '__main__':
    main()
