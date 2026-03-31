# build_image_features.py — Pre-computes CLIP image embeddings for every image
# in the dataset/ folder and saves them to disk so the search server can load
# them instantly at startup instead of re-encoding every time.

import torch                              # PyTorch deep-learning framework
import numpy as np                        # NumPy for array manipulation & saving .npy files
import faiss                              # Facebook AI Similarity Search – used here for L2 normalization
from PIL import Image                     # Pillow – opens image files into PIL Image objects
import clip                               # OpenAI CLIP library
from clip_model import model, preprocess, device   # Shared CLIP model, preprocessor, and device
import os                                 # OS utilities for file / directory operations

# Create the features/ directory if it doesn't already exist.
# This is where the computed embeddings and file paths will be saved.
os.makedirs("features", exist_ok=True)

# Only process files whose extension is a common image format.
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Accumulators: one list for feature vectors, one for corresponding file paths.
image_features = []   # Will hold each image's CLIP embedding (1-D numpy array)
image_paths = []      # Will hold the relative path string for each image

# Iterate over every file in the dataset/ directory.
for file in os.listdir("dataset"):
    # Skip non-image files (e.g. .txt, .json, sub-folders, etc.)
    if os.path.splitext(file)[1].lower() not in SUPPORTED_EXT:
        continue

    # Build the relative path used both for loading and later serving via the API.
    path = f"dataset/{file}"

    # preprocess() converts the PIL Image to a tensor the model expects.
    # unsqueeze(0) adds a batch dimension (shape: [1, 3, 224, 224]).
    # .to(device) moves the tensor to GPU/CPU to match the model.
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    # Encode the image into a 512-dim embedding vector without computing gradients
    # (no_grad saves memory and speeds up inference).
    with torch.no_grad():
        features = model.encode_image(image)
    
    # Move the embedding to CPU, convert to a flat numpy array, and store it.
    image_features.append(features.cpu().numpy().flatten())
    # Store the path so we can map an index back to its file later.
    image_paths.append(path)

# Stack all feature vectors into a single 2-D numpy array (N × 512), dtype float32
# so FAISS and later search operations work correctly.
image_features = np.array(image_features, dtype="float32")

# Normalize every feature vector to unit length (L2 norm = 1).
# After normalization, inner-product (dot product) equals cosine similarity,
# which is what the search index uses to rank results.
faiss.normalize_L2(image_features)

# Persist the normalized features and their matching paths to .npy files.
np.save("features/image_features.npy", image_features)   # shape: (N, 512)
np.save("features/paths.npy", image_paths)                # shape: (N,)