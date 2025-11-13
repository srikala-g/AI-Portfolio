#!/usr/bin/env python3
import os
import io
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def build_and_save_mappings(mapping_path):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets library not available to build mappings: " + str(e))

    print("Building mappings from 'huggan/wikiart' via datasets (may download metadata)...")
    ds = load_dataset("huggan/wikiart", split="train", trust_remote_code=True)
    mappings = {
        "artist": ds.features["artist"].names if "artist" in ds.features else [],
        "genre":  ds.features["genre"].names if "genre" in ds.features else [],
        "style":  ds.features["style"].names if "style" in ds.features else []
    }

    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)
    print(f"Saved mappings to {mapping_path}")
    return mappings

def load_mappings(mapping_path):
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r") as f:
                mappings = json.load(f)
            print(f"Loaded mappings from {mapping_path}")
            return mappings
        except Exception as e:
            print(f"⚠️ Failed to read mapping file {mapping_path}: {e} — rebuilding.")
    return build_and_save_mappings(mapping_path)

def safe_lookup(mapping, key, index):
    try:
        if key not in mapping or not isinstance(mapping[key], list):
            return None
        if index < 0 or index >= len(mapping[key]):
            return None
        return mapping[key][index]
    except Exception:
        return None

def save_image_from_bytes(image_data, output_dir, index):
    image_name = f"image_{index:05d}.jpg"
    image_path = os.path.join(output_dir, image_name)
    try:
        if isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
        else:
            image_bytes = image_data  # assume raw bytes

        with Image.open(io.BytesIO(image_bytes)) as img:
            img.convert("RGB").save(image_path, "JPEG")
        return image_name, image_path
    except Exception as e:
        print(f"⚠️ Failed to save image {image_name}: {e}")
        return None, None

# --------------------------------------------------
# Main function
# --------------------------------------------------
def create_json_and_download_images(parquet_dir, json_path, image_dir, mapping_path):
    parquet_dir = Path(parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    print(f"📦 Found {len(parquet_files)} parquet files.")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Load or build mappings
    mappings = load_mappings(mapping_path)

    records = []
    idx_counter = 0

    for parquet_file in parquet_files:
        print(f"\n📄 Processing {parquet_file}")
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"⚠️ Failed to read {parquet_file}: {e}")
            continue

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
            try:
                image_data = row.get("image")
                if image_data is None:
                    continue

                artist_id = int(row.get("artist", -1))
                genre_id  = int(row.get("genre", -1))
                style_id  = int(row.get("style", -1))

                artist_name = safe_lookup(mappings, "artist", artist_id)
                genre_name  = safe_lookup(mappings, "genre", genre_id)
                style_name  = safe_lookup(mappings, "style", style_id)

                image_name, image_path = save_image_from_bytes(image_data, image_dir, idx_counter)
                if not image_name:
                    continue

                record = {
                    "artist": artist_id,
                    "genre": genre_id,
                    "style": style_id,
                    "artist_name": artist_name,
                    "genre_name": genre_name,
                    "style_name": style_name,
                    "image_name": image_name,
                    "image_url": os.path.relpath(image_path, os.path.dirname(json_path)).replace(os.sep, "/")
                }
                records.append(record)
                idx_counter += 1

            except Exception as e:
                print(f"⚠️ Error processing record: {e}")
                continue

    # Write JSON output
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Saved {len(records)} records to {json_path}")
    print(f"🖼️ Images stored in: {image_dir}")

# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSON and extract images from all parquet files.")
    parser.add_argument("--parquet-dir", required=True, help="Directory containing parquet files.")
    parser.add_argument("--json", required=True, help="Output JSON file path.")
    parser.add_argument("--images", required=True, help="Directory to save images.")
    parser.add_argument(
        "--mapping",
        default="/Users/srikala/projects/AI-Portfolio/art_identifier/data/dataset/wikiart/metadata_mappings.json",
        help="Path to mapping JSON (artist/genre/style names). Will be auto-built if missing."
    )

    args = parser.parse_args()
    create_json_and_download_images(args.parquet_dir, args.json, args.images, args.mapping)
