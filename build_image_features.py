import torch
import numpy as np
import faiss
from PIL import Image
import clip
from clip_model import model, preprocess, device
import os

os.makedirs("features", exist_ok=True)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

image_features = []
image_paths = []

for file in os.listdir("dataset"):
    if os.path.splitext(file)[1].lower() not in SUPPORTED_EXT:
        continue
    path = f"dataset/{file}"

    image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image)
    
    image_features.append(features.cpu().numpy().flatten())
    image_paths.append(path)

image_features = np.array(image_features, dtype="float32")
faiss.normalize_L2(image_features)

np.save("features/image_features.npy", image_features)
np.save("features/paths.npy", image_paths)