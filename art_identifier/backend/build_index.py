"""
Build ChromaDB index for artworks with batch processing to avoid memory issues.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import chromadb
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Build ChromaDB index for artworks")
    parser.add_argument("--metadata", required=True, type=str, help="Path to metadata JSON file")
    parser.add_argument("--images", required=True, type=str, help="Path to folder containing images")
    parser.add_argument("--chroma-db-path", default="./chroma_db", type=str, help="Path to ChromaDB directory")
    parser.add_argument("--collection-name", default="artworks", type=str, help="ChromaDB collection name")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size for embedding images")
    return parser.parse_args()


def main(metadata_path, images_dir, chroma_db_path, collection_name, batch_size=32):
    metadata_path = Path(metadata_path)
    images_dir = Path(images_dir)
    chroma_db_path = Path(chroma_db_path)

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} metadata entries.")

    # Initialize CLIP
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(chroma_db_path))
    collection = client.get_or_create_collection(collection_name)

    print("Encoding images and preparing embeddings in batches...")

    batch_images = []
    batch_ids = []
    batch_metadatas = []

    for idx, item in enumerate(tqdm(metadata)):
        image_name = item.get("image_name")
        if not image_name:
            print(f"⚠️ No image filename key found for metadata index {idx}, skipping.")
            continue

        image_path = images_dir / image_name
        if not image_path.exists():
            print(f"⚠️ Image file {image_path} not found, skipping.")
            continue

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Failed to open image {image_path}: {e}, skipping.")
            continue

        batch_images.append(image)
        batch_ids.append(str(idx))
        # Store minimal metadata for retrieval
        batch_metadatas.append({
            "title": item.get("artist_name", "Unknown"),
            "artist": item.get("artist_name", "Unknown"),
            "genre": item.get("genre_name", ""),
            "style": item.get("style_name", ""),
            "description": item.get("artist_description", ""),
            "image_path": str(image_path)
        })

        # If batch is full, process embeddings
        if len(batch_images) == batch_size:
            add_batch(collection, batch_images, batch_ids, batch_metadatas, model, processor, device)
            batch_images, batch_ids, batch_metadatas = [], [], []

    # Add remaining images
    if batch_images:
        add_batch(collection, batch_images, batch_ids, batch_metadatas, model, processor, device)

    print(f"✅ Finished building ChromaDB collection '{collection_name}' with {len(collection.get()['ids'])} items.")


def add_batch(collection, images, ids, metadatas, model, processor, device):
    # Encode batch
    inputs = processor(images=images, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    embeddings /= embeddings.norm(dim=-1, keepdim=True)
    embeddings = embeddings.cpu().numpy()

    # Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        metadata_path=args.metadata,
        images_dir=args.images,
        chroma_db_path=args.chroma_db_path,
        collection_name=args.collection_name,
        batch_size=args.batch_size
    )
