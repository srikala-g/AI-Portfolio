import os
import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import chromadb

class ArtworkSearcher:
    def __init__(
        self,
        chroma_db_path: str,
        metadata_json_path: str,
        images_dir: str
    ):
        """
        Initialize ArtworkSearcher.
        Args:
            chroma_db_path (str): Path to ChromaDB directory
            metadata_json_path (str): Path to wiki_art_data.json
            images_dir (str): Path to images folder
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.images_dir = Path(images_dir)
        self.metadata_json_path = Path(metadata_json_path)

        # Load metadata
        if not self.metadata_json_path.exists():
            raise FileNotFoundError(f"Metadata JSON not found at {self.metadata_json_path}")
        with open(self.metadata_json_path, "r") as f:
            self.metadata = json.load(f)
        print(f"✅ Loaded {len(self.metadata)} metadata records.")

        # Load CLIP model
        print("🔹 Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Connect to ChromaDB
        print(f"🔹 Connecting to ChromaDB at {self.chroma_db_path}...")
        self.client = chromadb.PersistentClient(path=str(self.chroma_db_path))
        self.collection = self.client.get_or_create_collection("wikiart")
        print("✅ ArtworkSearcher initialized.")

    # -----------------------------
    # Embedding Helpers
    # -----------------------------
    def _encode_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    # -----------------------------
    # Search Methods
    # -----------------------------
    def search_by_text(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        query_emb = self._encode_text(query).tolist()[0]
        results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
        return self._format_results(results)

    def search_by_image(self, image_path: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        img = Image.open(image_path).convert("RGB")
        query_emb = self._encode_image(img).tolist()[0]
        results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
        return self._format_results(results)

    def search_by_image_and_text(self, image_path: str, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        img = Image.open(image_path).convert("RGB")
        img_emb = self._encode_image(img)
        txt_emb = self._encode_text(query_text)
        combined_emb = (img_emb + txt_emb) / 2
        results = self.collection.query(query_embeddings=[combined_emb.tolist()[0]], n_results=top_k)
        return self._format_results(results)

    # -----------------------------
    # Format Results
    # -----------------------------
    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not results or not results.get("ids"):
            return []

        formatted = []
        ids = results["ids"][0]
        distances = results.get("distances", [[0.0]*len(ids)])[0]

        for i, item_id in enumerate(ids):
            # Handle IDs like "shard1_0"
            if isinstance(item_id, str) and "_" in item_id:
                idx = int(item_id.split("_")[-1])
            elif isinstance(item_id, int):
                idx = item_id
            else:
                idx = None

            meta = {}
            if idx is not None and 0 <= idx < len(self.metadata):
                meta = self.metadata[idx]

            image_name = meta.get("image_name") or meta.get("image_url") or ""
            image_url = f"/images/{image_name}" if image_name else None

            formatted.append({
                "id": str(item_id),
                "title": meta.get("artist_name", "Unknown"),
                "artist": meta.get("artist_name", "Unknown"),
                "genre": meta.get("genre_name", "Unknown Genre"),
                "style": meta.get("style_name", "Unknown Style"),
                "description": meta.get("description") or meta.get("artist_description", ""),
                "image_url": image_url,
                "similarity_score": round(1 - distances[i], 4),
            })

        return formatted
