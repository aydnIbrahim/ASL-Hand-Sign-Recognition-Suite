import os, math, json, time
from datetime import datetime
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# -----------------------------------------------------------------------------
# ResNet‑18 ASL trainer
# -----------------------------------------------------------------------------

class AddGaussianNoise:
    """Additive Gaussian noise (expects tensor)."""
    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma
    def __call__(self, tensor):
        if not torch.is_tensor(tensor):
            tensor = transforms.functional.to_tensor(tensor)
        return torch.clamp(tensor + torch.randn_like(tensor) * self.sigma, 0.0, 1.0)


def resnet18_gray(num_classes: int, pretrained: bool = False):
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_balanced_sampler(targets):
    cnt = Counter(targets)
    weights = [1.0 / cnt[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    ROOT = "data_processed"
    TRAIN_DIR = os.path.join(ROOT, "train")
    VAL_DIR = os.path.join(ROOT, "valid") if os.path.isdir(os.path.join(ROOT, "valid")) else os.path.join(ROOT, "test")

    BATCH = 64
    EPOCHS = 20
    LR_MAX = 3e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ---- stats ----
    with open(os.path.join(ROOT, "stats.json")) as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]
    print(f"Device: {DEVICE} | mean={mean:.4f} std={std:.4f}", flush=True)

    # ---- transforms ----
    base_resize = [transforms.Resize((128, 128)), transforms.Grayscale(num_output_channels=1)]

    train_tf = transforms.Compose([
        *base_resize,
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        AddGaussianNoise(0.03),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        transforms.Normalize([mean], [std]),
    ])

    val_tf = transforms.Compose([
        *base_resize,
        transforms.ToTensor(),
        transforms.Normalize([mean], [std]),
    ])

    # ---- datasets ----
    print("Loading datasets...", flush=True)
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)
    num_classes = len(train_ds.classes)
    print(f"  → Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Classes: {num_classes}", flush=True)

    # ---- dataloaders ----
    sampler = create_balanced_sampler(train_ds.targets)
    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    # ---- model & optim ----
    model = resnet18_gray(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=len(train_dl), pct_start=0.3, div_factor=10, final_div_factor=10)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"runs/asl_{run_id}")

    print("Starting training...", flush=True)

    best_val = math.inf
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===", flush=True)
        epoch_start = time.time()

        # ---- train ----
        model.train()
        tr_loss, tr_correct = 0.0, 0
        for x, y in tqdm(train_dl, desc="Train", unit="batch"):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                y_hat = model(x)
                loss = criterion(y_hat, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tr_loss += loss.item() * x.size(0)
            tr_correct += (y_hat.argmax(1) == y).sum().item()

        train_loss = tr_loss / len(train_ds)
        train_acc = tr_correct / len(train_ds)

        # ---- validate ----
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            for x, y in tqdm(val_dl, desc="Val  ", unit="batch"):
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (y_hat.argmax(1) == y).sum().item()
        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalars("acc", {"train": train_acc, "val": val_acc}, epoch + 1)

        print(f"Epoch {epoch+1}: train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} | time {(time.time()-epoch_start):.1f}s", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            ckpt = f"asl_resnet18_best.pth"
            torch.save(model.state_dict(), ckpt)
            print("  ↳ saved best checkpoint", ckpt, flush=True)

    writer.close()
    print("Training finished!", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
