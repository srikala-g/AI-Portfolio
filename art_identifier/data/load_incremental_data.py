"""
Load incremental data from parquet files and append to wiki_art_data.json.
"""

import json
from pathlib import Path
from datasets import load_dataset
from typing import Set, Tuple

# Paths
DATA_DIR = Path(__file__).parent
PARQUET_FILE = DATA_DIR / "dataset" / "wikiart" / "data" / "train-00001-of-00072.parquet"
OUTPUT_JSON = DATA_DIR / "wiki_art_data.json"
DATASET_INFO = DATA_DIR / "dataset" / "wikiart" / "dataset_infos.json"


def load_label_mappings():
    """Load label mappings from dataset_infos.json."""
    with open(DATASET_INFO, 'r') as f:
        dataset_info = json.load(f)
    
    features = dataset_info['huggan--wikiart']['features']
    return {
        'artist_names': features['artist']['names'],
        'genre_names': features['genre']['names'],
        'style_names': features['style']['names']
    }


def get_genre_descriptions():
    """Get genre descriptions."""
    return {
        "abstract_painting": "Abstract painting focuses on shapes, colors, forms, and gestural marks rather than depicting recognizable objects or scenes.",
        "cityscape": "Cityscape paintings depict urban environments, buildings, streets, and city life.",
        "genre_painting": "Genre painting portrays scenes from everyday life, showing ordinary people engaged in common activities.",
        "illustration": "Illustration art is created to accompany, explain, or decorate text, often found in books, magazines, or advertisements.",
        "landscape": "Landscape paintings depict natural scenery such as mountains, valleys, trees, rivers, and forests.",
        "nude_painting": "Nude painting focuses on the unclothed human figure as the primary subject.",
        "portrait": "Portrait painting captures the likeness, personality, and mood of a person or group of people.",
        "religious_painting": "Religious painting depicts scenes, figures, or themes from religious texts, mythology, or spiritual traditions.",
        "sketch_and_study": "Sketch and study works are preliminary drawings or paintings made in preparation for a finished work.",
        "still_life": "Still life paintings depict inanimate objects such as flowers, fruit, tableware, and other everyday items arranged in a composition.",
        "Unknown Genre": "Genre classification is unknown or not applicable."
    }


def get_style_descriptions():
    """Get style descriptions."""
    return {
        "Abstract_Expressionism": "Abstract Expressionism emphasizes spontaneous, automatic, or subconscious creation, with emphasis on the act of painting itself.",
        "Action_painting": "Action painting is a style of abstract expressionism where paint is spontaneously dribbled, splashed, or smeared onto the canvas.",
        "Analytical_Cubism": "Analytical Cubism breaks down objects into geometric forms and analyzes them from multiple viewpoints simultaneously.",
        "Art_Nouveau": "Art Nouveau is characterized by flowing lines, organic forms, and decorative patterns inspired by nature.",
        "Baroque": "Baroque art is dramatic, emotional, and uses intense light and shadow contrasts to create a sense of movement and tension.",
        "Color_Field_Painting": "Color Field Painting focuses on large areas of flat, solid color spread across the canvas to create a meditative effect.",
        "Contemporary_Realism": "Contemporary Realism depicts subjects in a realistic manner using modern techniques and contemporary themes.",
        "Cubism": "Cubism represents subjects from multiple viewpoints simultaneously, breaking them into geometric shapes and fragments.",
        "Early_Renaissance": "Early Renaissance art emphasizes naturalism, perspective, and humanism, marking the transition from medieval to modern art.",
        "Expressionism": "Expressionism distorts reality to express emotional experience rather than physical reality.",
        "Fauvism": "Fauvism uses bold, non-naturalistic colors and simplified forms to express emotion.",
        "High_Renaissance": "High Renaissance art achieves perfect balance, harmony, and ideal beauty through mastery of perspective and human anatomy.",
        "Impressionism": "Impressionism captures the fleeting effects of light and color, often painted en plein air with visible brushstrokes.",
        "Mannerism_Late_Renaissance": "Mannerism emphasizes artificiality, elegance, and sophisticated compositions with elongated figures and unusual colors.",
        "Minimalism": "Minimalism uses simple geometric forms, neutral colors, and minimal elements to create art with maximum impact.",
        "Naive_Art_Primitivism": "Naive Art features childlike simplicity, flat perspectives, and bold colors, often created by self-taught artists.",
        "New_Realism": "New Realism depicts contemporary subjects with photographic precision and attention to detail.",
        "Northern_Renaissance": "Northern Renaissance art focuses on detailed realism, rich colors, and meticulous attention to surface textures.",
        "Pointillism": "Pointillism uses small, distinct dots of color applied in patterns to form an image when viewed from a distance.",
        "Pop_Art": "Pop Art incorporates imagery from popular culture, advertising, and mass media, often using bright colors and bold graphics.",
        "Post_Impressionism": "Post-Impressionism extends Impressionism while rejecting its limitations, emphasizing geometric forms and symbolic content.",
        "Realism": "Realism depicts subjects truthfully and accurately without idealization or romanticization.",
        "Rococo": "Rococo art is characterized by lightness, elegance, and elaborate ornamentation with playful, whimsical themes.",
        "Romanticism": "Romanticism emphasizes emotion, individualism, and the sublime beauty of nature and dramatic scenes.",
        "Symbolism": "Symbolism uses symbolic imagery and metaphors to represent ideas, emotions, and spiritual concepts.",
        "Synthetic_Cubism": "Synthetic Cubism combines different materials and textures to create collaged compositions with simplified forms.",
        "Ukiyo_e": "Ukiyo-e is a Japanese art style featuring woodblock prints depicting scenes from everyday life, landscapes, and theater."
    }


def get_artist_description(artist_name_raw: str) -> str:
    """Get artist description."""
    famous_artists = {
        "vincent-van-gogh": "Dutch Post-Impressionist painter known for bold colors, dramatic brushwork, and emotional intensity.",
        "pablo-picasso": "Spanish artist and co-founder of Cubism, one of the most influential artists of the 20th century.",
        "claude-monet": "French Impressionist painter known for his series of paintings depicting light and atmosphere.",
        "leonardo-da-vinci": "Italian Renaissance polymath, painter, and inventor, creator of the Mona Lisa and The Last Supper.",
        "salvador-dali": "Spanish Surrealist artist known for bizarre and dreamlike imagery.",
        "rembrandt": "Dutch Golden Age painter and etcher, master of light and shadow.",
        "michelangelo": "Italian Renaissance sculptor, painter, and architect, creator of the Sistine Chapel ceiling.",
        "henri-matisse": "French artist known for his use of color and fluid draughtsmanship, leader of Fauvism.",
        "edgar-degas": "French Impressionist artist known for paintings, sculptures, and drawings of dancers and racehorses.",
        "paul-cezanne": "French Post-Impressionist painter who laid the foundations for modern art.",
        "pierre-auguste-renoir": "French Impressionist painter known for his vibrant, light-filled scenes of people.",
        "gustav-klimt": "Austrian symbolist painter known for his decorative style and use of gold leaf.",
        "andy-warhol": "American artist and leading figure in the Pop Art movement, known for his silkscreen prints.",
        "francisco-goya": "Spanish romantic painter and printmaker, known for his dark and dramatic works.",
        "edvard-munch": "Norwegian Expressionist painter, creator of The Scream.",
        "henri-de-toulouse-lautrec": "French Post-Impressionist painter known for depicting Parisian nightlife.",
        "paul-gauguin": "French Post-Impressionist artist known for his use of bold colors and exotic subjects.",
        "Unknown Artist": "Artist information is unknown or not available."
    }
    
    if artist_name_raw in famous_artists:
        return famous_artists[artist_name_raw]
    
    clean_name = artist_name_raw.replace("-", " ").title()
    return f"{clean_name} is an artist whose works are part of the WikiArt collection."


def get_existing_combinations(existing_data: list) -> Set[Tuple[int, int, int]]:
    """Get set of existing (artist, genre, style) combinations."""
    combinations = set()
    for record in existing_data:
        if all(k in record for k in ['artist', 'genre', 'style']):
            combinations.add((
                int(record['artist']),
                int(record['genre']),
                int(record['style'])
            ))
    return combinations


def create_record(
    artist_id: int,
    genre_id: int,
    style_id: int,
    image_idx: int,
    label_mappings: dict,
    genre_descriptions: dict,
    style_descriptions: dict
) -> dict:
    """Create a new record in the format matching existing JSON."""
    artist_names = label_mappings['artist_names']
    genre_names = label_mappings['genre_names']
    style_names = label_mappings['style_names']
    
    # Get raw names
    artist_name_raw = artist_names[artist_id]
    genre_name_raw = genre_names[genre_id]
    style_name_raw = style_names[style_id]
    
    # Format names
    artist_name = artist_name_raw.replace("-", " ").title()
    genre_name = genre_name_raw.replace("_", " ").title()
    style_name = style_name_raw.replace("_", " ").title()
    
    # Create title
    if style_name and style_name != 'Unknown':
        title = f"{style_name} {genre_name} by {artist_name}"
    else:
        title = f"{genre_name} by {artist_name}"
    
    return {
        "artist": artist_id,
        "genre": genre_id,
        "style": style_id,
        "image_name": f"image_{image_idx:05d}.jpg",
        "image_url": f"images/image_{image_idx:05d}.jpg",
        "artist_description": get_artist_description(artist_name_raw),
        "genre_description": genre_descriptions.get(
            genre_name_raw,
            f"{genre_name} is a genre classification for artworks."
        ),
        "style_description": style_descriptions.get(
            style_name_raw,
            f"{style_name} is an artistic style or movement."
        ),
        "artist_name": artist_name,
        "genre_name": genre_name,
        "style_name": style_name,
        "artist_name_raw": artist_name_raw,
        "genre_name_raw": genre_name_raw,
        "style_name_raw": style_name_raw,
        "title": title
    }


def main():
    """Main function to load incremental data."""
    print("=" * 60)
    print("Loading Incremental Data from Parquet File")
    print("=" * 60)
    
    # Check if parquet file exists
    if not PARQUET_FILE.exists():
        print(f"❌ Error: Parquet file not found at {PARQUET_FILE}")
        return
    
    # Load existing JSON data
    print(f"\n📂 Loading existing data from {OUTPUT_JSON}...")
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, 'r') as f:
            existing_data = json.load(f)
        print(f"   ✅ Loaded {len(existing_data)} existing records")
    else:
        existing_data = []
        print(f"   ⚠️  No existing file found, starting fresh")
    
    # Get existing combinations to avoid duplicates
    existing_combinations = get_existing_combinations(existing_data)
    print(f"   📊 Found {len(existing_combinations)} unique (artist, genre, style) combinations")
    
    # Load label mappings
    print(f"\n📋 Loading label mappings from {DATASET_INFO}...")
    label_mappings = load_label_mappings()
    genre_descriptions = get_genre_descriptions()
    style_descriptions = get_style_descriptions()
    print(f"   ✅ Loaded mappings for {len(label_mappings['artist_names'])} artists, "
          f"{len(label_mappings['genre_names'])} genres, {len(label_mappings['style_names'])} styles")
    
    # Load dataset from HuggingFace (using the parquet file path as reference)
    print(f"\n📦 Loading dataset from HuggingFace...")
    print(f"   Note: Using HuggingFace datasets library to load data")
    try:
        dataset = load_dataset("huggan/wikiart", split="train")
        print(f"   ✅ Dataset loaded: {len(dataset)} total examples")
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return
    
    # Process dataset and find new records
    print(f"\n🔍 Processing dataset to find new records...")
    new_records = []
    start_idx = len(existing_data)
    processed = 0
    skipped = 0
    
    for idx, example in enumerate(dataset):
        processed += 1
        
        artist_id = int(example['artist'])
        genre_id = int(example['genre'])
        style_id = int(example['style'])
        
        combination = (artist_id, genre_id, style_id)
        
        # Skip if combination already exists
        if combination in existing_combinations:
            skipped += 1
            if processed % 1000 == 0:
                print(f"   ⏳ Processed {processed:,}/{len(dataset):,} examples "
                      f"(found {len(new_records):,} new, skipped {skipped:,} duplicates)")
            continue
        
        # Create new record
        image_idx = start_idx + len(new_records)
        new_record = create_record(
            artist_id, genre_id, style_id, image_idx,
            label_mappings, genre_descriptions, style_descriptions
        )
        
        new_records.append(new_record)
        existing_combinations.add(combination)  # Mark as added
        
        # Progress update
        if len(new_records) % 100 == 0:
            print(f"   ✅ Found {len(new_records):,} new records so far...")
    
    print(f"\n📊 Processing Summary:")
    print(f"   Total examples processed: {processed:,}")
    print(f"   New records found: {len(new_records):,}")
    print(f"   Duplicates skipped: {skipped:,}")
    
    # Append new records if any
    if new_records:
        print(f"\n💾 Appending {len(new_records):,} new records to {OUTPUT_JSON}...")
        existing_data.extend(new_records)
        
        print(f"   Saving updated JSON file...")
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"\n✅ Successfully completed!")
        print(f"   📈 Total records before: {start_idx:,}")
        print(f"   📈 Total records after: {len(existing_data):,}")
        print(f"   ➕ Added: {len(new_records):,} new records")
    else:
        print(f"\n✅ No new records to add - all combinations already exist in the JSON file.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

