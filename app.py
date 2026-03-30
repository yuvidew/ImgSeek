from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from search_text import search_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

@app.get("/search")
def search(query: str, k: int = 5):
    results = search_text(query, k)
    return {"results": results}