import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from search import ArtworkSearcher
from generate_backstory import BackstoryGenerator
from PIL import Image
import io

# Load environment variables from .env file
# Try loading from backend directory first, then parent directory
env_paths = [
    Path(__file__).parent / ".env",  # backend/.env
    Path(__file__).parent.parent / ".env",  # art_identifier/.env
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
        break
else:
    # Fallback: try loading from current directory
    load_dotenv()

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
# Initialize backstory generator
# --------------------------------------------------
backstory_generator = None

def get_backstory_generator():
    """Lazy initialization of backstory generator."""
    global backstory_generator
    if backstory_generator is None:
        try:
            backstory_generator = BackstoryGenerator(
                chroma_db_path=CHROMA_DB_DIR,
                metadata_json_path=METADATA_JSON
            )
        except Exception as e:
            print(f"Warning: Could not initialize BackstoryGenerator: {e}")
            print("Backstory generation will not be available. Set OPENAI_API_KEY environment variable.")
    return backstory_generator

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

@app.get("/artwork/{artwork_id}/backstory")
async def get_artwork_backstory(artwork_id: str):
    """
    Generate a backstory for an artwork.
    
    Args:
        artwork_id: ID of the artwork (e.g., "shard1_0" or numeric index)
    
    Returns:
        JSON response with backstory and metadata
    """
    generator = get_backstory_generator()
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Backstory generation is not available. Please set OPENAI_API_KEY environment variable."
        )
    
    try:
        result = generator.generate_backstory(artwork_id)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating backstory: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to WikiArt Search API",
        "endpoints": [
            "/search/text",
            "/search/image",
            "/search/image-and-text",
            "/artwork/{artwork_id}/backstory"
        ]
    }
