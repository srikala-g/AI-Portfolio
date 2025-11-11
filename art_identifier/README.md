# Art Identifier

An AI-powered artwork identification and search system using embeddings. This application allows you to search through a curated dataset of artworks using text descriptions, image uploads, or a combination of both.

## Project Structure

```
art-identifier/
├── data/
│   ├── wiki_art_data.json      # Curated dataset (~200 items)
│   └── images/                  # Artwork images
├── backend/
│   ├── embeddings.py            # Embedding generation utilities
│   ├── build_index.py           # Build search index from data
│   ├── search.py                # Search functionality
│   ├── api.py                   # FastAPI application
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── app_streamlit.py         # Streamlit web interface
│   └── streamlit_requirements.txt
├── notebooks/
│   └── explore_embeddings.ipynb # Jupyter notebook for exploration
├── docker-compose.yml           # Docker Compose configuration
├── Dockerfile                   # Backend Docker image
├── Dockerfile.streamlit         # Frontend Docker image
└── README.md                    # This file
```

## Features

- **Text Search**: Search artworks by describing them in natural language
- **Image Search**: Upload an image to find similar artworks
- **Combined Search**: Use both image and text for more precise results
- **FastAPI Backend**: RESTful API for programmatic access
- **Streamlit Frontend**: User-friendly web interface
- **Docker Support**: Easy deployment with Docker Compose

## Setup

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional, for containerized deployment)

### Local Development

1. **Install Backend Dependencies**

```bash
cd backend
pip install -r requirements.txt
```

2. **Build the Search Index**

First, ensure you have:
- `data/wiki_art_data.json` with artwork metadata
- `data/images/` directory with corresponding images

Then build the index:

```bash
cd backend
python build_index.py --metadata ../data/wiki_art_data.json --images ../data/images --output artworks_index.pkl
```

3. **Start the Backend API**

```bash
cd backend
python api.py
# Or using uvicorn directly:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

4. **Start the Frontend**

In a new terminal:

```bash
cd frontend
pip install -r streamlit_requirements.txt
streamlit run app_streamlit.py
```

The frontend will be available at `http://localhost:8501`

### Docker Deployment

1. **Build and Start Services**

```bash
docker-compose up --build
```

This will start:
- Backend API at `http://localhost:8000`
- Frontend at `http://localhost:8501`

2. **Stop Services**

```bash
docker-compose down
```

## Usage

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Text Search
```bash
POST /search/text
Content-Type: application/json

{
  "query": "starry night painting",
  "top_k": 5
}
```

#### Image Search
```bash
POST /search/image
Content-Type: multipart/form-data

file: <image_file>
top_k: 5
```

#### Combined Search
```bash
POST /search/image-and-text
Content-Type: multipart/form-data

file: <image_file>
query: "impressionist style"
top_k: 5
```

### Frontend Interface

1. **Text Search Tab**: Enter a text description and get matching artworks
2. **Image Search Tab**: Upload an image to find similar artworks
3. **Combined Search Tab**: Upload an image and add text description for refined results

## Technical Details

### Embeddings

The system uses CLIP (Contrastive Language-Image Pre-training) model from `sentence-transformers`:
- **Image Embeddings**: Generated using CLIP's vision encoder
- **Text Embeddings**: Generated using CLIP's text encoder
- **Combined Embeddings**: Concatenation of image and text embeddings

### Search Algorithm

- Uses cosine similarity for finding similar artworks
- Normalizes embeddings for accurate similarity computation
- Returns top-k most similar results

### Data Format

The `wiki_art_data.json` file should contain records with:
- `artist`: Artist ID or name
- `genre`: Genre ID or name
- `style`: Style ID or name

The system converts these to human-readable metadata using label mappings from the dataset.

## Development

### Adding New Artworks

1. Add metadata entries to `data/wiki_art_data.json`
2. Add corresponding images to `data/images/`
3. Rebuild the index: `python backend/build_index.py`

### Customizing Embeddings

Modify `backend/embeddings.py` to:
- Use different models
- Change embedding dimensions
- Implement custom fusion strategies

### Exploring Embeddings

Use the Jupyter notebook `notebooks/explore_embeddings.ipynb` to:
- Visualize embeddings
- Analyze similarity distributions
- Experiment with different search strategies

## Troubleshooting

### Index Not Found

If you see "Index file not found" error:
1. Make sure you've run `build_index.py`
2. Check that `artworks_index.pkl` exists in the `backend/` directory

### Images Not Displaying

- Verify image paths in metadata match actual file locations
- Check that images are in `data/images/` directory
- Ensure image file formats are supported (JPG, PNG)

### API Connection Issues

- Verify backend is running on port 8000
- Check CORS settings if accessing from different origin
- Review API logs for error messages

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

