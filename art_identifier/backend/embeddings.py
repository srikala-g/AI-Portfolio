"""
Generate embeddings for artwork images and metadata.
"""

import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer


def load_label_mappings():
    """Load label mappings from dataset_infos.json."""
    dataset_info_path = Path("../data/dataset/wikiart/dataset_infos.json")
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    features = dataset_info['huggan--wikiart']['features']
    return {
        'artist_names': features['artist']['names'],
        'genre_names': features['genre']['names'],
        'style_names': features['style']['names']
    }


def convert_wiki_art_record(record: Dict[str, Any], idx: int, label_mappings: Dict) -> Dict[str, Any]:
    """Convert a wiki_art_data.json record to the expected metadata format."""
    artist_names = label_mappings['artist_names']
    genre_names = label_mappings['genre_names']
    style_names = label_mappings['style_names']
    
    artist_name = artist_names[record["artist"]].replace("-", " ").title()
    genre_name = genre_names[record["genre"]].replace("_", " ").title()
    style_name = style_names[record["style"]].replace("_", " ").title()
    
    return {
        "id": f"wikiart_{idx:05d}",
        "title": f"Artwork {idx + 1}",
        "artist": artist_name,
        "genre": genre_name,
        "style": style_name,
        "image_path": f"images/image_{idx:05d}.jpg",
        "description": f"A {genre_names[record['genre']].replace('_', ' ')} in the {style_names[record['style']].replace('_', ' ')} style by {artist_name}.",
        "year": "Unknown",
        "medium": "Unknown",
        "museum": "WikiArt"
    }


def load_artworks_metadata(metadata_path: str) -> List[Dict[str, Any]]:
    """
    Load artworks metadata from JSON file. 
    ONLY uses wiki_art_data.json - no other data sources.
    """
    metadata_file = Path(metadata_path)
    
    # Ensure we're only using wiki_art_data.json
    if "wiki_art_data" not in metadata_file.name.lower():
        raise ValueError(
            f"Only wiki_art_data.json is supported. "
            f"Received: {metadata_file.name}. "
            f"Please use data/wiki_art_data.json"
        )
    
    # Load wiki_art_data.json and convert
    with open(metadata_path, 'r', encoding='utf-8') as f:
        wiki_data = json.load(f)
    
    # Load label mappings
    label_mappings = load_label_mappings()
    
    # Convert each record
    artworks = []
    for idx, record in enumerate(wiki_data):
        artwork = convert_wiki_art_record(record, idx, label_mappings)
        artworks.append(artwork)
    
    return artworks


# Initialize models (lazy loading)
_image_model = None
_text_model = None

def _get_image_model():
    """Lazy load the image embedding model."""
    global _image_model
    if _image_model is None:
        # Use CLIP model from sentence-transformers for multimodal embeddings
        try:
            _image_model = SentenceTransformer('clip-ViT-B-32')
        except Exception as e:
            print(f"Warning: Could not load CLIP model: {e}. Using fallback.")
            _image_model = "fallback"
    return _image_model

def _get_text_model():
    """Lazy load the text embedding model."""
    global _text_model
    if _text_model is None:
        try:
            # Use CLIP for text as well to keep same embedding space as images
            # This ensures better compatibility between image and text embeddings
            _text_model = SentenceTransformer('clip-ViT-B-32')
        except Exception as e:
            print(f"Warning: Could not load text model: {e}. Using fallback.")
            _text_model = "fallback"
    return _text_model

def generate_image_embedding(image_path: str) -> np.ndarray:
    """
    Generate embedding for an artwork image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array representing the image embedding
    """
    model = _get_image_model()
    
    if model == "fallback":
        # Fallback to random if model not available
        return np.random.rand(512).astype(np.float32)
    
    try:
        # Load and encode image
        image = Image.open(image_path).convert('RGB')
        embedding = model.encode(image, convert_to_numpy=True)
        # Ensure it's float32 and the right shape
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"Error generating image embedding for {image_path}: {e}")
        # Return zero vector on error
        return np.zeros(512, dtype=np.float32)


def generate_text_embedding(text: str) -> np.ndarray:
    """
    Generate embedding for text description.
    
    Args:
        text: Text description of the artwork
        
    Returns:
        numpy array representing the text embedding
    """
    model = _get_text_model()
    
    if model == "fallback":
        # Fallback to random if model not available
        return np.random.rand(512).astype(np.float32)
    
    try:
        # Encode text using CLIP (same model as images for consistency)
        embedding = model.encode(text, convert_to_numpy=True)
        # Ensure it's float32 and the right shape
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        # Return zero vector on error
        return np.zeros(512, dtype=np.float32)


def combine_embeddings(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """
    Combine image and text embeddings into a single vector.
    
    Args:
        image_emb: Image embedding vector
        text_emb: Text embedding vector
        
    Returns:
        Combined embedding vector
    """
    # Simple concatenation (can be improved with learned fusion)
    return np.concatenate([image_emb, text_emb])


if __name__ == "__main__":
    # Example usage
    metadata = load_artworks_metadata("../data/wiki_art_data.json")
    print(f"Loaded {len(metadata)} artworks")
    if metadata:
        print(f"Sample: {metadata[0]}")

