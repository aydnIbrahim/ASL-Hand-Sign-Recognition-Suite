# 🤚📚 ASL Hand-Sign Recognition Suite

A full pipeline for **American Sign Language (ASL) static‑letter recognition** powered by **PyTorch**, **MediaPipe**, and **Streamlit**.

*End‑to‑end:* from raw image folders → preprocessing & augmentation → **ResNet‑18** training → quantitative evaluation → **real‑time webcam inference**.

---

## ✨ Key Features

| Stage             | Highlights                                                                                                                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data prep**     | ‑ One‑click `split_dataset.py` train/valid/test splitter <br>‑ `preprocessing.py` CLAHE + blur + 128×128 grayscale & per‑dataset **mean/std** auto‑compute <br>‑ Online augmentation (rotation, perspective, jitter, GaussianNoise, AutoAugment…) |
| **Training**      | ‑ Single‑channel **ResNet‑18** with balanced sampler <br>‑ **AdamW + OneCycleLR** <br>‑ Mixed precision (AMP) & TensorBoard logs                                                                                                                  |
| **Evaluation**    | ‑ `evaluate.py` gives accuracy, class report & confusion matrix PNG                                                                                                                                                                               |
| **Real‑time GUI** | ‑ `asl_gui_streamlit.py` <br>‑ **MediaPipe** hand detection & dynamic crop <br>‑ Live top‑5 probability bar chart & bbox overlay                                                                                                                  |

---

## 🗂️ Repository Structure

```
.
├─ data/                  # raw images (before preprocessing)
├─ data_processed/        # auto‑generated: processed & split images + stats.json
│   ├─ train/
│   ├─ valid/ or test/
│   └─ stats.json         # {"mean": …, "std": …}
├─ models/                # saved .pth checkpoints (optional)
├─ preprocessing.py       # disk preprocessing + augmentation definitions
├─ split_dataset.py       # train/valid split helper
├─ train.py               # ResNet‑18 trainer
├─ evaluate.py            # metrics & confusion matrix
├─ asl_gui_streamlit.py   # Streamlit real‑time app
└─ README.md
```

---

## ⚡ Quick Start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # see sample below
```

<details>
<summary>requirements.txt (minimal)</summary>

```text
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # or cpu
opencv-python
mediapipe
streamlit
tqdm
matplotlib
scikit-learn
pillow
```

</details>

### 2. Prepare Dataset

Place your raw ASL letter images in `data/<CLASS_NAME>/…` (PNG/JPG). `J` & `Z` **excluded** (they are motion gestures).

```bash
python split_dataset.py \
  --input_dir data --output_dir data --test_size 0.2
```

### 3. Preprocess & Augment

```bash
python preprocessing.py          # writes data_processed/ + stats.json
```

### 4. Train

```bash
python train.py                  # outputs asl_resnet18_best.pth
```

*TensorBoard:* `tensorboard --logdir runs` → [http://localhost:6006](http://localhost:6006)

### 5. Evaluate

```bash
python evaluate.py asl_resnet18_best.pth
```

Outputs:

* overall / per‑class precision‑recall‑F1
* `confusion_matrix.png`
* `eval_metrics.json`

### 6. Real‑Time Webcam Demo

```bash
streamlit run asl_gui_streamlit.py
```

* Sidebar → choose **Start** to open webcam.
* Green box & label show live prediction; right panel plots top‑5 probabilities.

---

## 🛠️ Customisation

### Hyper‑parameters (train.py)

* `BATCH`, `EPOCHS`, `LR_MAX`
* Augmentations – edit `train_tf` / `AUGMENT_TRAIN` in **preprocessing.py**.

### Adding Classes

1. Add new folder in `data/<NEW_CLASS>` with images.
2. Rerun **preprocessing**, **train**, **evaluate**.

### CPU‑only

All scripts detect CUDA automatically. For enforced CPU:

```bash
python evaluate.py asl_resnet18_best.pth --device cpu
```

---

## 📋 Troubleshooting

| Issue                                                          | Fix                                                                                                 |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| *`RuntimeError: Tried to instantiate class '__path__._path'…`* | Restart Streamlit after code edits; ensure **indentation** is correct.                              |
| Webcam black frame                                             | Lower `FRAME_SIZE` in `asl_gui_streamlit.py` or pick correct camera index in `cv2.VideoCapture(0)`. |
| Hand not detected                                              | Increase `min_detection_confidence` or ensure good lighting / background contrast.                  |

---

## 📜 License

[MIT](LICENSE)
