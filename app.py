# app.py — FastAPI web server that exposes the image-search API
# and serves dataset images as static files.

from fastapi import FastAPI                          # FastAPI framework for building the REST API
from fastapi.middleware.cors import CORSMiddleware   # Middleware to allow cross-origin requests
from fastapi.staticfiles import StaticFiles          # Serves static files (images) over HTTP
from search_text import search_text                  # The core text-to-image search function

# Create the FastAPI application instance.
app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing) so the front-end (index.html)
# served from a different origin/port can call this API without being blocked
# by the browser's same-origin policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Allow requests from any origin
    allow_methods=["*"],     # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],     # Allow all request headers
)

# Mount the dataset/ folder at the /dataset URL path so images can be
# accessed directly by the browser, e.g. http://localhost:8000/dataset/photo.jpg
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")


@app.get("/search")
def search(query: str, k: int = 5):
    """Handle GET /search?query=...&k=...

    Args:
        query (str): Natural-language description of the desired image.
        k (int):     Maximum number of results to return (default 5).

    Returns:
        dict: {"results": [ {"path": "dataset/...", "score": 0.xx}, ... ]}
    """
    # Delegate to search_text() which encodes the query via CLIP,
    # searches the FAISS index, and returns matching image paths + scores.
    results = search_text(query, k)
    return {"results": results}