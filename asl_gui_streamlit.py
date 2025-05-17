import os, json, time
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import transforms, models
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_PATH_DEFAULT = "asl_resnet18_best.pth"
DATA_ROOT_DEFAULT = "data_processed"  # must contain stats.json
FRAME_SIZE = 640  # webcam frame size for display

# -----------------------------------------------------------------------------
# Model factory (single‑channel ResNet‑18)
# -----------------------------------------------------------------------------

def resnet18_gray(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Pre‑processing helpers (identical to training)
# -----------------------------------------------------------------------------

def preprocess_hand_crop(crop_bgr: np.ndarray) -> Image.Image:
    """Gray + CLAHE + light Gaussian blur → PIL"""
    gray = (crop_bgr if len(crop_bgr.shape) == 2 else cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return Image.fromarray(denoised)

# -----------------------------------------------------------------------------
# Load model + transforms & cache
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_system(model_path: str, stats_path: str, class_names: List[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # stats
    with open(stats_path) as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std])
    ])

    model = resnet18_gray(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    return device, model, transform, hands

# -----------------------------------------------------------------------------
# Hand detection + crop util
# -----------------------------------------------------------------------------

def detect_first_hand(frame_rgb: np.ndarray, hands_detector) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return crop (BGR) and bbox (x1,y1,x2,y2) or (None, None)."""
    res = hands_detector.process(frame_rgb)
    if not res.multi_hand_landmarks:
        return None, None
    h, w, _ = frame_rgb.shape
    x_coords = [int(lm.x * w) for lm in res.multi_hand_landmarks[0].landmark]
    y_coords = [int(lm.y * h) for lm in res.multi_hand_landmarks[0].landmark]

    pad = 20
    x1, x2 = max(min(x_coords) - pad, 0), min(max(x_coords) + pad, w)
    y1, y2 = max(min(y_coords) - pad, 0), min(max(y_coords) + pad, h)
    crop_bgr = cv2.cvtColor(frame_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
    return crop_bgr, (x1, y1, x2, y2)

# -----------------------------------------------------------------------------
# Prediction helper
# -----------------------------------------------------------------------------

def predict(crop_img: Image.Image, model, transform, device):
    x = transform(crop_img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().squeeze()
    return probs

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="ASL Live Classifier", page_icon="🎥", layout="centered")
st.title("🤚🎥 ASL Real‑Time Letter Classifier")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model checkpoint (.pth)", MODEL_PATH_DEFAULT)
    data_root = st.text_input("Data root (stats + classes)", DATA_ROOT_DEFAULT)
    run = st.radio("Camera", ["Stop", "Start"], index=1)

# derive classes list
train_dir = os.path.join(data_root, "train")
CLASS_NAMES = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

if not (os.path.isfile(model_path) and os.path.isfile(os.path.join(data_root, "stats.json"))):
    st.error("Model or stats.json not found!")
    st.stop()

# load components
device, model, transform, hands_detector = load_system(model_path, os.path.join(data_root, "stats.json"), CLASS_NAMES)
st.success(f"Loaded model on {device}")

frame_placeholder = st.empty()
prob_placeholder = st.empty()

if run == "Start":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE)
    st.info("Press the sidebar → Stop to end the stream.")

    while run == "Start":
        success, frame_bgr = cap.read()
        if not success:
            st.error("Camera failure.")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        crop_bgr, bbox = detect_first_hand(frame_rgb, hands_detector)

        if crop_bgr is not None:
            pil_crop = preprocess_hand_crop(crop_bgr)
            probs = predict(pil_crop, model, transform, device)
            pred_idx = int(torch.argmax(probs))
            pred_cls = CLASS_NAMES[pred_idx]
            pred_conf = probs[pred_idx].item()

            # draw bbox & label
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_rgb, f"{pred_cls} {pred_conf*100:.1f}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                        # show probabilities bar
            topk_val, topk_idx = torch.topk(probs, 5)
            labels = [CLASS_NAMES[i] for i in topk_idx.cpu().tolist()][::-1]
            values = topk_val.cpu().tolist()[::-1]
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.barh(labels, values)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            prob_placeholder.pyplot(fig)
            plt.close(fig)
            plt.close(fig)
        else:
            prob_placeholder.info("No hand detected")

        frame_placeholder.image(frame_rgb, channels="RGB")

        # refresh flag (Streamlit re‑runs script on widget change)
        run = st.session_state.get("run", run)
        if run == "Stop":
            break

    cap.release()
    hands_detector.close()
    st.success("Stream stopped.")
else:
    st.info("Camera stopped. Use sidebar → Start to begin.")
