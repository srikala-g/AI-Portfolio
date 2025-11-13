#!/usr/bin/env python3
import os
import io
import json
import argparse
from tqdm import tqdm
import pandas as pd
from PIL import Image

# optional import to build mappings if missing
def build_and_save_mappings(mapping_path):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets library not available to build mappings: " + str(e))

    print("Building mappings from 'huggan/wikiart' via datasets (this may download metadata)...")
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
            print(f"⚠️ Failed to read mapping file {mapping_path}: {e} — will try to rebuild.")
    # Build if not present or failed to read
    return build_and_save_mappings(mapping_path)

def safe_lookup(mapping, key, index):
    try:
        if key not in mapping:
            return None
        names = mapping[key]
        if not isinstance(names, list):
            return None
        if index < 0 or index >= len(names):
            return None
        return names[index]
    except Exception:
        return None

def save_image_from_bytes(image_data, output_dir, index):
    image_name = f"image_{index:05d}.jpg"
    image_path = os.path.join(output_dir, image_name)
    try:
        if isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.convert("RGB").save(image_path, "JPEG")
            return image_name, image_path
        else:
            # image_data could be bytes directly
            if isinstance(image_data, (bytes, bytearray)):
                with Image.open(io.BytesIO(image_data)) as img:
                    img.convert("RGB").save(image_path, "JPEG")
                return image_name, image_path
        return None, None
    except Exception as e:
        print(f"⚠️ Failed to save image {image_name}: {e}")
        return None, None

def create_json_and_download_images(parquet_path, json_path, image_dir, mapping_path):
    # 1) validate inputs
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    print(f"📦 Loading parquet file: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet file {parquet_path}: {e}")

    print(f"✅ Loaded {len(df)} records.")
    # quick schema inspect
    print("Columns:", df.columns.tolist())
    if len(df) > 0:
        print("First row (preview):")
        # print only non-bytes previews to avoid huge dump
        preview = df.head(1).to_dict(orient="records")[0]
        for k,v in preview.items():
            if k == "image":
                print("  image: <image dict preview>")
            else:
                print(f"  {k}: {v}")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Load or build mappings
    mappings = load_mappings(mapping_path)

    records = []
    total = len(df)
    print(f"Processing {total} records and saving images to: {image_dir}")
    for idx, row in tqdm(df.iterrows(), total=total, desc="Processing records"):
        try:
            image_data = row.get("image")
            artist_id = int(row.get("artist", -1)) if row.get("artist") is not None else -1
            genre_id  = int(row.get("genre", -1)) if row.get("genre") is not None else -1
            style_id  = int(row.get("style", -1)) if row.get("style") is not None else -1

            # map ids to names if mapping available
            artist_name = safe_lookup(mappings, "artist", artist_id)
            genre_name  = safe_lookup(mappings, "genre", genre_id)
            style_name  = safe_lookup(mappings, "style", style_id)

            image_name, image_path = save_image_from_bytes(image_data, image_dir, idx)
            if not image_name:
                # skip records without images
                continue

            record = {
                "artist": artist_id,
                "genre": genre_id,
                "style": style_id,
                "artist_name": artist_name,
                "genre_name": genre_name,
                "style_name": style_name,
                "image_name": image_name,
                # image_url relative to json location (use simple images/... path)
                "image_url": os.path.relpath(image_path, os.path.dirname(json_path)).replace(os.sep, "/")
            }
            records.append(record)

        except Exception as e:
            print(f"⚠️ Error processing record index {idx}: {e}")
            continue

    # Write JSON output
    try:
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to write JSON to {json_path}: {e}")

    print(f"\n✅ Done. Saved {len(records)} records to {json_path}")
    print(f"🖼️ Images stored in: {image_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract JSON and images from WikiArt parquet shard.")
    parser.add_argument("--parquet", required=True, help="Path to parquet file.")
    parser.add_argument("--json", required=True, help="Output JSON file path.")
    parser.add_argument("--images", required=True, help="Directory to save images.")
    parser.add_argument(
        "--mapping",
        default="/Users/srikala/projects/AI-Portfolio/art_identifier/data/dataset/wikiart/metadata_mappings.json",
        help="Path to mapping JSON (artist/genre/style names). Will be auto-built if missing."
    )

    args = parser.parse_args()
    try:
        create_json_and_download_images(
            args.parquet,
            args.json,
            args.images,
            args.mapping
        )
    except Exception as exc:
        print("\n❌ Processing failed with error:")
        print(exc)
        print("\nPlease paste the above traceback/output here and I'll help further.")
        raise
