#!/usr/bin/env python3
"""
Script to extract titles from artwork images and add them to the JSON metadata.
Uses vision models to analyze images and extract titles.
"""

import json
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import time
from tqdm import tqdm

# Try to import OpenAI for vision API
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI not available. Will use OCR fallback.")

# Try to import pytesseract for OCR fallback
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("⚠️  pytesseract not available. OCR fallback disabled.")


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_title_with_vision(image_path: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Extract title from artwork image using GPT-4 Vision API.
    
    Args:
        image_path: Path to the image file
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
    
    Returns:
        Extracted title or None if extraction fails
    """
    if not HAS_OPENAI:
        return None
    
    try:
        # Get API key from parameter, env var, or raise error
        if api_key:
            key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass --api-key")
        
        client = openai.OpenAI(api_key=key)
        
        base64_image = encode_image(image_path)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an art expert. Look at this artwork image and extract the title of the artwork. The title is usually written on the artwork itself (often at the bottom), or you can identify it from the artwork's content. Return ONLY the title text, nothing else. If you cannot find a title, return 'Unknown'."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the title of this artwork? Extract the title if it's visible on the image, or identify it from the artwork's characteristics. Return only the title text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200
        )
        
        title = response.choices[0].message.content.strip()
        # Clean up the title
        title = title.replace('"', '').replace("'", "").strip()
        if title.lower() in ['unknown', 'none', 'n/a', 'not found', '']:
            return None
        return title if title else None
        
    except Exception as e:
        print(f"Error extracting title with vision API: {e}")
        return None


def extract_title_with_ocr(image_path: str) -> Optional[str]:
    """
    Extract title from artwork image using OCR.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Extracted title or None if extraction fails
    """
    if not HAS_OCR:
        return None
    
    try:
        img = Image.open(image_path)
        # Try to extract text from the bottom portion where titles are often located
        width, height = img.size
        # Focus on bottom 20% of image where titles are often written
        bottom_region = img.crop((0, int(height * 0.8), width, height))
        
        # Extract text
        text = pytesseract.image_to_string(bottom_region, config='--psm 6')
        text = text.strip()
        
        # Clean and filter text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            # Take the first substantial line as potential title
            title = lines[0]
            if len(title) > 3 and len(title) < 200:  # Reasonable title length
                return title
        
        return None
        
    except Exception as e:
        print(f"Error extracting title with OCR: {e}")
        return None


def extract_title(image_path: str, use_vision: bool = True, api_key: Optional[str] = None) -> Optional[str]:
    """
    Extract title from artwork image, trying vision API first, then OCR.
    
    Args:
        image_path: Path to the image file
        use_vision: Whether to use vision API (default: True)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
    
    Returns:
        Extracted title or None if extraction fails
    """
    # Try vision API first if available
    if use_vision and HAS_OPENAI:
        title = extract_title_with_vision(image_path, api_key)
        if title:
            return title
    
    # Fallback to OCR
    if HAS_OCR:
        title = extract_title_with_ocr(image_path)
        if title:
            return title
    
    return None


def process_artworks(
    json_path: str,
    images_dir: str,
    output_path: Optional[str] = None,
    batch_size: int = 100,
    use_vision: bool = True,
    api_key: Optional[str] = None,
    start_index: int = 0,
    max_items: Optional[int] = None
):
    """
    Process all artworks and add titles to JSON.
    
    Args:
        json_path: Path to input JSON file
        images_dir: Directory containing artwork images
        output_path: Path to output JSON file (default: overwrites input)
        batch_size: Number of items to process before saving
        use_vision: Whether to use vision API
        api_key: OpenAI API key
        start_index: Index to start processing from (for resuming)
        max_items: Maximum number of items to process (None for all)
    """
    # Load JSON
    print(f"📖 Loading JSON from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        artworks = json.load(f)
    
    total = len(artworks)
    print(f"✅ Loaded {total} artworks")
    
    if output_path is None:
        output_path = json_path
    
    images_dir_path = Path(images_dir)
    
    # Check how many already have titles
    already_processed = sum(1 for art in artworks if 'title' in art and art['title'])
    print(f"📊 {already_processed} artworks already have titles")
    
    # Process artworks
    processed = 0
    failed = 0
    end_index = min(start_index + max_items, total) if max_items else total
    
    print(f"🚀 Processing artworks {start_index} to {end_index}...")
    
    for idx in tqdm(range(start_index, end_index), desc="Processing"):
        artwork = artworks[idx]
        
        # Skip if already has a title
        if 'title' in artwork and artwork.get('title'):
            continue
        
        # Get image path
        image_name = artwork.get('image_name')
        if not image_name:
            artwork['title'] = None
            failed += 1
            continue
        
        image_path = images_dir_path / image_name
        
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            artwork['title'] = None
            failed += 1
            continue
        
        # Extract title
        try:
            title = extract_title(str(image_path), use_vision=use_vision, api_key=api_key)
            artwork['title'] = title
            if title:
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Error processing {image_name}: {e}")
            artwork['title'] = None
            failed += 1
        
        # Save periodically
        if (idx + 1) % batch_size == 0:
            print(f"💾 Saving progress... ({idx + 1}/{end_index})")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(artworks, f, indent=2, ensure_ascii=False)
            time.sleep(0.1)  # Small delay to avoid rate limits
    
    # Final save
    print(f"💾 Saving final results...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(artworks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Processing complete!")
    print(f"   - Successfully extracted titles: {processed}")
    print(f"   - Failed/No title found: {failed}")
    print(f"   - Total processed: {processed + failed}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract titles from artwork images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: Processing 81,444 images with GPT-4 Vision API will be expensive (~$500-1000+).
Consider processing in batches or using --max to limit the number of images.

Example usage:
  # Test with 10 images first
  python extract_titles.py --max 10
  
  # Process all images (requires OPENAI_API_KEY env var)
  python extract_titles.py
  
  # Resume from index 1000
  python extract_titles.py --start 1000
        """
    )
    parser.add_argument("--json", type=str, 
                       default="/Users/srikala/projects/AI-Portfolio/art_identifier/data/json/wiki_art_data.json",
                       help="Path to JSON file")
    parser.add_argument("--images", type=str,
                       default="/Users/srikala/projects/AI-Portfolio/art_identifier/data/images",
                       help="Path to images directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON path (default: overwrites input)")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for saving progress")
    parser.add_argument("--no-vision", action="store_true",
                       help="Disable vision API, use OCR only")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--start", type=int, default=0,
                       help="Start index (for resuming)")
    parser.add_argument("--max", type=int, default=None,
                       help="Maximum number of items to process")
    
    args = parser.parse_args()
    
    # Check API key before starting
    if not args.no_vision and HAS_OPENAI:
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ ERROR: OpenAI API key required for vision extraction")
            print("   Set OPENAI_API_KEY environment variable or use --api-key")
            print("   Or use --no-vision to use OCR only (requires pytesseract)")
            exit(1)
        
        # Estimate cost
        with open(args.json, 'r') as f:
            total = len(json.load(f))
        items_to_process = args.max if args.max else (total - args.start)
        # Rough estimate: $0.01-0.015 per image for GPT-4o
        estimated_cost = items_to_process * 0.012
        print(f"💰 Estimated cost for {items_to_process} images: ~${estimated_cost:.2f}")
        response = input("Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            exit(0)
    
    process_artworks(
        json_path=args.json,
        images_dir=args.images,
        output_path=args.output,
        batch_size=args.batch_size,
        use_vision=not args.no_vision,
        api_key=args.api_key,
        start_index=args.start,
        max_items=args.max
    )

