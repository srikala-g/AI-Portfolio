"""
Append new artwork data to an existing ChromaDB collection safely.
Supports enriched WikiArt JSON metadata format with artist_name, genre_name, style_name, title.
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
    parser = argparse.ArgumentParser(description="Append new artworks to ChromaDB collection")
    parser.add_argument("--metadata", required=True, type=str, help="Path to metadata JSON file")
    parser.add_argument("--images", required=True, type=str, help="Path to folder containing images")
    parser.add_argument("--chroma-db-path", default="./chroma_db", type=str, help="Path to ChromaDB directory")
    parser.add_argument("--collection-name", default="wikiart", type=str, help="ChromaDB collection name")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size for embedding images")
    return parser.parse_args()


def main(metadata_path, images_dir, chroma_db_path, collection_name, batch_size=32):
    metadata_path = Path(metadata_path)
    images_dir = Path(images_dir)
    chroma_db_path = Path(chroma_db_path)

    # Load metadata JSON
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(f"📖 Loaded {len(metadata)} metadata records from {metadata_path.name}")

    # Initialize CLIP model
    print("🚀 Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(chroma_db_path))
    collection = client.get_or_create_collection(collection_name)
    print(f"🏗️  Appending to collection '{collection_name}' at {chroma_db_path}")

    # Track batch
    batch_images, batch_ids, batch_metadatas = [], [], []
    total_added = 0

    # Start indexing
    for idx, item in enumerate(tqdm(metadata, desc="Indexing images")):
        image_name = item.get("image_name")
        if not image_name:
            print(f"⚠️ Missing 'image_name' for record {idx}, skipping.")
            continue

        image_path = images_dir / image_name
        if not image_path.exists():
            print(f"⚠️ Image not found: {image_path}, skipping.")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Failed to open image {image_path}: {e}")
            continue

        batch_images.append(image)
        batch_ids.append(f"append_{idx}")  # unique ID for new records
        batch_metadatas.append({
            "title": item.get("title", ""),
            "artist_name": item.get("artist_name", "Unknown"),
            "genre_name": item.get("genre_name", ""),
            "style_name": item.get("style_name", ""),
            "artist_description": item.get("artist_description", ""),
            "genre_description": item.get("genre_description", ""),
            "style_description": item.get("style_description", ""),
            "image_path": str(image_path)
        })

        if len(batch_images) >= batch_size:
            total_added += add_batch(collection, batch_images, batch_ids, batch_metadatas, model, processor, device)
            batch_images, batch_ids, batch_metadatas = [], [], []

    # Add remaining images
    if batch_images:
        total_added += add_batch(collection, batch_images, batch_ids, batch_metadatas, model, processor, device)

    print(f"\n✅ Finished appending. {total_added} new artworks added to '{collection_name}'.")


def add_batch(collection, images, ids, metadatas, model, processor, device):
    # Prepare images for CLIP
    inputs = processor(images=images, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)

    embeddings = embeddings.cpu().numpy()
    collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)
    return len(ids)


if __name__ == "__main__":
    args = parse_args()
    main(
        metadata_path=args.metadata,
        images_dir=args.images,
        chroma_db_path=args.chroma_db_path,
        collection_name=args.collection_name,
        batch_size=args.batch_size
    )
