import numpy as np
import faiss
import torch
import clip 
from clip_model import model, preprocess, device

# Load data
features = np.load("features/image_features.npy")
paths = np.load("features/paths.npy")

# Build FAISS index (inner product = cosine similarity on normalized vectors)
index = faiss.IndexFlatIP(features.shape[1])
index.add(features)

def search_text(query, k=5):
    try:
        text = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy().astype("float32")
        
        scores, indices = index.search(text_features, k)
        threshold = 0.23
        results = [
            {"path": str(paths[i]), "score": float(scores[0][j])}
            for j, i in enumerate(indices[0])
            if scores[0][j] >= threshold
        ]
        return results
    except Exception as e:
        print(f"Error: {e}")
        return []