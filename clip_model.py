# clip_model.py — Loads the CLIP model once so it can be shared across other modules.

import clip    # OpenAI's CLIP library for vision-language tasks
import torch   # PyTorch – the deep-learning framework CLIP is built on

# Select GPU ("cuda") when available; otherwise fall back to CPU.
# Using a GPU dramatically speeds up encoding images and text.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP ViT-B/32 model and its matching image preprocessor.
#   model       – the neural network that encodes images & text into a shared embedding space.
#   preprocess  – a torchvision transform pipeline (resize, crop, normalize, etc.)
#                 that converts a PIL Image into the tensor format the model expects.
model, preprocess = clip.load("ViT-B/32", device=device)