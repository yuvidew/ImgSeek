# search_text.py — Provides the text-to-image search function.
# It loads pre-computed image embeddings, builds a FAISS similarity index,
# and exposes search_text() which encodes a text query with CLIP and finds
# the most similar images.

import numpy as np                                     # NumPy for loading .npy feature files
import faiss                                           # FAISS for fast nearest-neighbor search
import torch                                           # PyTorch (needed by CLIP for inference)
import clip                                            # OpenAI CLIP – tokenizes the text query
from clip_model import model, preprocess, device       # Shared CLIP model & device

# --- Load pre-computed data (runs once at module import time) ---

# features: (N, 512) float32 array – one normalized CLIP embedding per image.
features = np.load("features/image_features.npy")
# paths: (N,) array of strings – file paths corresponding to each feature row.
paths = np.load("features/paths.npy")

# Build a FAISS inner-product index.
# Because the image features are already L2-normalized (unit vectors),
# inner product == cosine similarity, giving us a fast cosine-similarity search.
# features.shape[1] is the embedding dimension (512).
index = faiss.IndexFlatIP(features.shape[1])
# Add all image embeddings to the index so they can be queried.
index.add(features)


def search_text(query, k=5):
    """Search for images that best match a natural-language query.

    Args:
        query (str): A text description, e.g. "a dog playing in the park".
        k (int):     Maximum number of results to return (default 5).

    Returns:
        list[dict]: Each dict has:
            - "path"  (str):   relative file path of the matching image.
            - "score" (float): cosine-similarity score (higher = better match).
          Only results with score >= threshold are included.
    """
    try:
        # Tokenize the query string into CLIP's input format and move to device.
        text = clip.tokenize([query]).to(device)

        # Encode the text into a 512-dim embedding (no gradient needed for inference).
        with torch.no_grad():
            text_features = model.encode_text(text)

            # Normalize to unit length so dot product equals cosine similarity.
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Move to CPU and cast to float32 (FAISS requires float32 numpy arrays).
            text_features = text_features.cpu().numpy().astype("float32")
        
        # Query the FAISS index for the top-k most similar image embeddings.
        # scores: (1, k) array of similarity values.
        # indices: (1, k) array of corresponding row indices in `features`.
        scores, indices = index.search(text_features, k)

        # Minimum similarity score an image must have to be included in results.
        # This filters out low-confidence / irrelevant matches.
        threshold = 0.23

        # Build the result list, keeping only matches above the threshold.
        results = [
            {"path": str(paths[i]), "score": float(scores[0][j])}
            for j, i in enumerate(indices[0])
            if scores[0][j] >= threshold
        ]
        return results

    except Exception as e:
        # Log any unexpected error and return an empty result set.
        print(f"Error: {e}")
        return []