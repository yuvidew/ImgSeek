# ImgSeek

A semantic image search engine that lets you find images using natural language queries. Powered by OpenAI's [CLIP](https://github.com/openai/CLIP) model and [FAISS](https://github.com/facebookresearch/faiss) for fast similarity search.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-orange)

## How It Works

1. **Feature Extraction** — Each image in the dataset is encoded into a 512-dimensional vector using CLIP's `ViT-B/32` vision encoder.
2. **Indexing** — Image embeddings are L2-normalized and stored in a FAISS inner-product index for cosine similarity search.
3. **Search** — A text query is encoded with CLIP's text encoder, normalized, and matched against all image embeddings. Results above a similarity threshold are returned.

## Project Structure

```
ImgSeek/
├── app.py                    # FastAPI server with search endpoint
├── build_image_features.py   # Extracts & saves CLIP embeddings for all images
├── clip_model.py             # Loads CLIP model (shared across modules)
├── search_text.py            # Text-to-image search using FAISS
├── index.html                # Frontend UI
├── dataset/                  # Place your images here
└── features/                 # Auto-generated embeddings & paths
    ├── image_features.npy
    └── paths.npy
```

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision faiss-cpu pillow numpy fastapi uvicorn
pip install git+https://github.com/openai/CLIP.git
```

### 2. Add Images

Place your images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`) in the `dataset/` folder.

### 3. Build Image Features

```bash
python build_image_features.py
```

This encodes all images and saves the normalized feature vectors to `features/`.

### 4. Start the Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### 5. Open the UI

Open `index.html` in your browser and start searching.

## API

### `GET /search`

| Parameter | Type   | Default | Description              |
|-----------|--------|---------|--------------------------|
| `query`   | string | —       | Natural language query    |
| `k`       | int    | 5       | Max number of results     |

**Example:**

```
GET http://127.0.0.1:8000/search?query=a+dog+playing&k=5
```

**Response:**

```json
{
  "results": [
    { "path": "dataset/dog.jpg", "score": 0.27 },
    { "path": "dataset/puppy.png", "score": 0.25 }
  ]
}
```

## Tech Stack

- **CLIP (ViT-B/32)** — Vision-language model for encoding images and text into a shared embedding space
- **FAISS** — Fast approximate nearest neighbor search
- **FastAPI** — Backend API server
- **HTML/CSS/JS** — Minimal frontend

## License

MIT
