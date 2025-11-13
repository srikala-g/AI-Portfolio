import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from search import ArtworkSearcher
from PIL import Image
import io

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_DIR = "/Users/srikala/projects/AI-Portfolio/art_identifier/data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
METADATA_JSON = os.path.join(DATA_DIR, "json", "wiki_art_data.json")

# --------------------------------------------------
# Initialize FastAPI app
# --------------------------------------------------
app = FastAPI(title="WikiArt Semantic Search API", version="1.3")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# --------------------------------------------------
# Pydantic models
# --------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    n_results: int = 10

# --------------------------------------------------
# Initialize searcher
# --------------------------------------------------
searcher = ArtworkSearcher(
    chroma_db_path=CHROMA_DB_DIR,
    metadata_json_path=METADATA_JSON,
    images_dir=IMAGES_DIR
)

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "collection": "wikiart"}

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.post("/search/text")
async def search_text(request: SearchRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = searcher.search_by_text(request.query, request.n_results)
    return JSONResponse(content=results)

@app.post("/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = Query(10)):
    try:
        contents = await file.read()
        Image.open(io.BytesIO(contents))  # validate image
        temp_path = os.path.join("/tmp", file.filename)
        with open(temp_path, "wb") as f:
            f.write(contents)
        results = searcher.search_by_image(temp_path, top_k)
        os.remove(temp_path)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

@app.post("/search/image-and-text")
async def search_image_and_text(
    file: UploadFile = File(...),
    query: str = Form(""),
    top_k: int = Form(10)
):
    try:
        contents = await file.read()
        Image.open(io.BytesIO(contents))  # validate
        temp_path = os.path.join("/tmp", file.filename)
        with open(temp_path, "wb") as f:
            f.write(contents)
        final_query = query or "artwork"
        results = searcher.search_by_image_and_text(temp_path, final_query, top_k)
        os.remove(temp_path)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image+text: {e}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to WikiArt Search API",
        "endpoints": ["/search/text", "/search/image", "/search/image-and-text"]
    }
