import os, json, argparse, itertools
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Helper: single‑channel ResNet‑18 (same as in training)
# -----------------------------------------------------------------------------

def resnet18_gray(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# -----------------------------------------------------------------------------
# Confusion‑matrix plotting
# -----------------------------------------------------------------------------

def plot_confusion(cm, class_names, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # text annotations
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Evaluation routine
# -----------------------------------------------------------------------------

def evaluate(model_path: str, data_root: str = "data_processed", batch: int = 64, device: str = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ----- load stats -----
    with open(os.path.join(data_root, "stats.json")) as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    # ----- test transform -----
    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std]),
    ])

    test_dir = os.path.join(data_root, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"{test_dir} not found! Make sure you have a test split.")

    test_ds = datasets.ImageFolder(test_dir, transform=tf)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=0)
    class_names = test_ds.classes

    # ----- model -----
    model = resnet18_gray(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for x, y in tqdm(test_dl, desc="Evaluating"):
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu()
            y_pred.extend(preds.numpy())
            y_true.extend(y.numpy())

    # metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    acc = report["accuracy"]
    print(f"\nOverall accuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(os.path.dirname(model_path), "confusion_matrix.png")
    plot_confusion(cm, class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # save metrics json
    json_path = os.path.join(os.path.dirname(model_path), "eval_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Detailed metrics saved to {json_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained ASL ResNet‑18 model")
    parser.add_argument("model", help="Path to .pth checkpoint", nargs="?", default="asl_resnet18_best.pth")
    parser.add_argument("--data_root", default="data_processed", help="Root directory containing stats.json and test/ folder")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    args = parser.parse_args()
    evaluate(args.model, args.data_root, args.batch, args.device)
