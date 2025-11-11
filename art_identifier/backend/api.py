"""
FastAPI application for artwork identification service.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import tempfile
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search import ArtworkSearcher

app = FastAPI(
    title="Art Identifier API",
    description="API for identifying and searching artworks using embeddings",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Lazy-loaded searcher
# -----------------------------
_searcher: ArtworkSearcher = None

def get_searcher() -> ArtworkSearcher:
    global _searcher
    if _searcher is None:
        chroma_db_path = Path(__file__).parent / "chroma_db"
        if not chroma_db_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"ChromaDB not found at {chroma_db_path}. Please run build_index.py first."
            )
        _searcher = ArtworkSearcher(db_path=str(chroma_db_path))
    return _searcher

# -----------------------------
# Request/Response Models
# -----------------------------
class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    id: str
    title: str
    artist: str
    genre: Optional[str] = None
    style: Optional[str] = None
    description: Optional[str] = None
    image_path: Optional[str] = None
    similarity_score: float

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {
        "message": "Art Identifier API",
        "version": "1.0.0",
        "endpoints": {
            "text_search": "/search/text",
            "image_search": "/search/image",
            "image_and_text_search": "/search/image-and-text",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        searcher = get_searcher()
        return {
            "status": "healthy",
            "artworks_loaded": len(searcher.metadata)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/search/text", response_model=List[SearchResult])
async def search_by_text(request: TextSearchRequest):
    try:
        searcher = get_searcher()
        results = searcher.search_by_text(request.query, top_k=request.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image", response_model=List[SearchResult])
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = 5
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        try:
            searcher = get_searcher()
            results = searcher.search_by_image(tmp_path, top_k=top_k)
            return results
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image-and-text", response_model=List[SearchResult])
async def search_by_image_and_text(
    file: UploadFile = File(...),
    query: str = "",
    top_k: int = 5
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        try:
            searcher = get_searcher()
            results = searcher.search_by_image_and_text(tmp_path, query, top_k=top_k)
            return results
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
