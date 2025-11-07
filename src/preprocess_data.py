import os
import pandas as pd
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse

# -------------------------
# 1. Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# Use ResNet50 with updated weights syntax
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # remove final FC layer
resnet.eval().to(device)

# -------------------------
# 2. Image transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -------------------------
# 3. Feature extraction
# -------------------------
def extract_features(image_path, output_dir):
    """Extract feature from one image and save it as .npy"""
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet(img_tensor).squeeze().cpu().numpy()
        np.save(output_dir / f"{Path(image_path).stem}.jpg.npy", feat)
        return True
    except Exception as e:
        print(f"⚠️ Error with {image_path}: {e}")
        return False

def extract_image_features(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0

    for img_path in tqdm(df["image_path"], desc="Extracting features"):
        if extract_features(img_path, output_dir):
            success += 1

    print(f"✅ Features saved in: {output_dir}")
    print(f"✅ Successfully processed {success}/{len(df)} images")

# -------------------------
# 4. Main script
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/splits/train_split.csv",
                        help="Path to CSV split file")
    parser.add_argument("--out", type=str, default="data/features",
                        help="Output directory for .npy files")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of samples for quick testing (0 = all)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    # Load CSV
    print(f"📦 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Keep only existing image files
    df = df[df["image_path"].apply(lambda p: Path(p).exists())]
    print(f"✅ Found {len(df)} valid image paths")

    # Optional: limit number of rows
    if args.limit > 0:
        print(f"⚡ Limiting to {args.limit} images for quick test")
        df = df.head(args.limit)

    # Extract and save features
    extract_image_features(df, out_dir)
