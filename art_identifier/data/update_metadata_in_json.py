import json
import argparse

def load_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        return json.load(f)

def update_json(json_path, output_path, mapping_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    mapping = load_mapping(mapping_path)
    artists = mapping.get("artist", {})
    genres = mapping.get("genre", {})
    styles = mapping.get("style", {})

    for item in data:
        if "artist" in item:
            item["artist_name"] = artists.get(str(item["artist"]), "Unknown")
        if "genre" in item:
            item["genre_name"] = genres.get(str(item["genre"]), "Unknown")
        if "style" in item:
            item["style_name"] = styles.get(str(item["style"]), "Unknown")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Updated JSON saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to the existing JSON file")
    parser.add_argument("--mapping", required=True, help="Path to the metadata mapping JSON")
    parser.add_argument("--output", required=True, help="Path to save updated JSON file")
    args = parser.parse_args()

    update_json(args.json, args.output, args.mapping)
