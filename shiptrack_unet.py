"""
Ship track segmentation using a lightweight U-Net on MODIS satellite imagery.

Dataset: CloudTracks (Chaudhry et al., 2024) — false-colour MODIS composites
with LabelMe polyline annotations of ship tracks.

Architecture: Two-level U-Net encoder–decoder with skip connections.

Usage:
    python shiptrack_unet.py train       # train on data/train and save weights
    python shiptrack_unet.py test        # run inference on data/test and save plots
"""

import os
import sys
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader

TRACK_WIDTH = 10
IMG_SIZE = 512
RESULTS_DIR = "results"
WEIGHTS_PATH = os.path.join(RESULTS_DIR, "shiptrack_unet.pth")
THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShipTrackDataset(Dataset):
    """Pairs MODIS PNG images with LabelMe JSON annotations and rasterises
    ship-track polylines into binary segmentation masks."""

    def __init__(self, img_dir, ann_dir, img_size=IMG_SIZE, limit=None):
        self.img_size = img_size
        self.pairs = self._find_pairs(img_dir, ann_dir, limit)

    def _find_pairs(self, img_dir, ann_dir, limit):
        jsons = sorted(f for f in os.listdir(ann_dir) if f.endswith(".json"))
        if limit:
            jsons = jsons[:limit]
        pairs = []
        for jf in jsons:
            stem = os.path.splitext(jf)[0]
            img_path = os.path.join(img_dir, stem + ".png")
            if os.path.isfile(img_path):
                pairs.append((img_path, os.path.join(ann_dir, jf)))
        if not pairs:
            raise FileNotFoundError(
                f"No matching image/annotation pairs in {img_dir} and {ann_dir}"
            )
        return pairs

    def _rasterise_tracks(self, annotation, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        for shape in annotation.get("shapes", []):
            if shape.get("label") != "shiptrack":
                continue
            pts = shape.get("points", [])
            if len(pts) < 2:
                continue
            coords = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [coords], False, 1, TRACK_WIDTH)
        return mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, ann_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        with open(ann_path) as f:
            annotation = json.load(f)
        mask = self._rasterise_tracks(annotation, h, w)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.fromarray(mask).resize(
            (self.img_size, self.img_size), Image.NEAREST
        )

        img_t = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)
        mask_t = torch.from_numpy(
            np.array(mask, dtype=np.float32)
        ).unsqueeze(0)

        return img_t, mask_t


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two consecutive Conv-BN-ReLU layers."""
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    """Lightweight two-level U-Net for binary segmentation."""

    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.bridge = ConvBlock(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out_conv = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bridge(self.pool(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs=10, batch_size=2, lr=1e-3, max_images=10,
          data_root="data/train"):

    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ds = ShipTrackDataset(
        os.path.join(data_root, "images"),
        os.path.join(data_root, "jsons"),
        limit=max_images,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            optimiser.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimiser.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(ds)
        print(f"epoch {epoch}/{epochs}, loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)

    print(f"saved to {WEIGHTS_PATH}")


# ---------------------------------------------------------------------------
# Testing / Visualisation
# ---------------------------------------------------------------------------

def test(data_root="data/test"):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ds = ShipTrackDataset(
        os.path.join(data_root, "images"),
        os.path.join(data_root, "jsons"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    print(f"loaded {WEIGHTS_PATH}, testing on {len(ds)} images")

    with torch.no_grad():
        for i in range(len(ds)):
            img_t, mask_t = ds[i]
            logits = model(img_t.unsqueeze(0).to(device))
            pred = torch.sigmoid(logits)[0, 0].cpu().numpy()

            img_np = img_t.permute(1, 2, 0).numpy()
            gt_np = mask_t.squeeze(0).numpy()
            pred_np = (pred > THRESHOLD).astype(np.float32)

            fig, ax = plt.subplots(1, 3, figsize=(14, 4))
            ax[0].imshow(img_np)
            ax[0].set_title("Input")
            ax[1].imshow(gt_np, cmap="gray")
            ax[1].set_title("Ground Truth")
            ax[2].imshow(pred_np, cmap="gray")
            ax[2].set_title("Prediction")
            for a in ax:
                a.axis("off")

            plt.tight_layout()
            out = os.path.join(RESULTS_DIR, f"result_{i + 1}.png")
            plt.savefig(out, dpi=150)
            plt.close(fig)
            print(f"saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
    test()
