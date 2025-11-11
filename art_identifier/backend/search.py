"""
Artwork search engine using ChromaDB and CLIP embeddings.
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import chromadb
from pathlib import Path
import json
import os


class ArtworkSearcher:
    def __init__(self, db_path: str, collection_name: str = "artworks"):
        """
        Initialize ArtworkSearcher.

        Args:
            db_path (str): Path to the ChromaDB directory.
            collection_name (str): Name of the Chroma collection.
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        # Derive metadata path relative to the project root
        project_root = self.db_path.parent.parent
        self.metadata_path = project_root / "data" / "wiki_art_data.json"
        if not self.metadata_path.exists():
            raise RuntimeError(f"Metadata file not found at {self.metadata_path}")

        print(f"🔹 Loading ChromaDB from: {self.db_path}")
        print(f"🔹 Loading metadata from: {self.metadata_path}")

        # Initialize CLIP model
        print("🔹 Loading CLIP model and processor...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Initialize Chroma client
        print("🔹 Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(self.collection_name)

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        print(f"✅ ArtworkSearcher initialized with {len(self.metadata)} artworks.")

    # -----------------------------
    # Embedding Helpers
    # -----------------------------
    def _encode_text(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    def _encode_image(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    # -----------------------------
    # Search Methods
    # -----------------------------
    def search_by_text(self, query: str, top_k: int = 5):
        """Search artworks by text."""
        if not query:
            return []
        print(f"🔍 Searching by text: {query}")
        query_embedding = self._encode_text(query).tolist()[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return self._format_results(results)

    def search_by_image(self, image_path: str, top_k: int = 5):
        """Search artworks by image path."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return []
        print(f"🔍 Searching by image: {os.path.basename(image_path)}")
        query_embedding = self._encode_image(image).tolist()[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return self._format_results(results)

    def search_by_image_and_text(self, image_path: str, query_text: str, top_k: int = 5):
        """Search by both image and text."""
        image = Image.open(image_path).convert("RGB")
        img_emb = self._encode_image(image)
        txt_emb = self._encode_text(query_text)
        combined_emb = (img_emb + txt_emb) / 2
        results = self.collection.query(query_embeddings=[combined_emb.tolist()[0]], n_results=top_k)
        return self._format_results(results)

    # -----------------------------
    # Format Results
    # -----------------------------
    def _format_results(self, results):
        formatted = []
        if not results or not results.get("ids"):
            return formatted

        for i, item_id in enumerate(results["ids"][0]):
            meta = {}
            # Validate and convert item_id to int if possible
            if isinstance(item_id, int) or (isinstance(item_id, str) and item_id.isdigit()):
                idx = int(item_id)
                if idx < len(self.metadata):
                    meta = self.metadata[idx]

            formatted.append({
                "id": str(item_id),
                "title": str(meta.get("title") or "Unknown"),
                "artist": str(meta.get("artist_name") or "Unknown"),
                "genre": str(meta.get("genre_name") or ""),
                "style": str(meta.get("style_name") or ""),
                "description": str(meta.get("description") or meta.get("artist_description") or ""),
                "image_path": str(meta.get("image_url") or meta.get("image_name") or ""),
                "similarity_score": float(1 - results["distances"][0][i]) if "distances" in results else 0.0,
            })
        return formatted

