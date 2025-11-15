"""
Generate concise backstory for artworks using LLM prompts based on metadata and embeddings.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# Load environment variables from .env file
# Try loading from backend directory first, then parent directory
env_paths = [
    Path(__file__).parent / ".env",  # backend/.env
    Path(__file__).parent.parent / ".env",  # art_identifier/.env
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    # Fallback: try loading from current directory
    load_dotenv()


class BackstoryGenerator:
    """Generate backstories for artworks using LLM based on metadata and similar artworks."""
    
    def __init__(
        self,
        chroma_db_path: str,
        metadata_json_path: str,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize BackstoryGenerator.
        
        Args:
            chroma_db_path: Path to ChromaDB directory
            metadata_json_path: Path to wiki_art_data.json
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.metadata_json_path = Path(metadata_json_path)
        
        # Load metadata
        if not self.metadata_json_path.exists():
            raise FileNotFoundError(f"Metadata JSON not found at {self.metadata_json_path}")
        with open(self.metadata_json_path, "r") as f:
            self.metadata = json.load(f)
        
        # Connect to ChromaDB for finding similar artworks
        self.client = chromadb.PersistentClient(path=str(self.chroma_db_path))
        self.collection = self.client.get_or_create_collection("wikiart")
        
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key parameter."
            )
        self.client_openai = OpenAI(api_key=api_key)
        
        print("✅ BackstoryGenerator initialized.")
    
    def _get_artwork_metadata(self, artwork_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific artwork by ID."""
        # Handle IDs like "shard1_0" or numeric indices
        if isinstance(artwork_id, str) and "_" in artwork_id:
            try:
                idx = int(artwork_id.split("_")[-1])
            except ValueError:
                return None
        elif isinstance(artwork_id, str) and artwork_id.isdigit():
            idx = int(artwork_id)
        elif isinstance(artwork_id, int):
            idx = artwork_id
        else:
            return None
        
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return None
    
    def _find_similar_artworks(self, artwork_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar artworks using embeddings from ChromaDB."""
        try:
            # Get the artwork's embedding from ChromaDB
            artwork_data = self.collection.get(ids=[artwork_id], include=["embeddings"])
            if not artwork_data or not artwork_data.get("ids") or not artwork_data.get("embeddings"):
                return []
            
            # Get the embedding vector
            artwork_embedding = artwork_data["embeddings"][0]
            
            # Query for similar artworks (we'll filter out the artwork itself later)
            query_results = self.collection.query(
                query_embeddings=[artwork_embedding],
                n_results=top_k + 1  # +1 to account for the artwork itself
            )
            
            similar = []
            if query_results and query_results.get("ids") and len(query_results["ids"]) > 0:
                for i, item_id in enumerate(query_results["ids"][0]):
                    if item_id == artwork_id:
                        continue  # Skip the artwork itself
                    
                    meta = self._get_artwork_metadata(item_id)
                    if meta:
                        similar.append({
                            "artist": meta.get("artist_name", "Unknown"),
                            "genre": meta.get("genre_name", "Unknown"),
                            "style": meta.get("style_name", "Unknown"),
                        })
                    
                    if len(similar) >= top_k:
                        break
            
            return similar
        except Exception as e:
            print(f"Warning: Could not find similar artworks: {e}")
            return []
    
    def _build_prompt(self, metadata: Dict[str, Any], similar_artworks: List[Dict[str, Any]]) -> str:
        """Build the LLM prompt for generating backstory."""
        artist = metadata.get("artist_name", "Unknown Artist")
        genre = metadata.get("genre_name", "Unknown Genre")
        style = metadata.get("style_name", "Unknown Style")
        description = metadata.get("description", "")
        
        # Build context about the artwork
        artwork_context = f"""
Artwork Information:
- Artist: {artist}
- Genre: {genre}
- Style: {style}
"""
        
        if description:
            artwork_context += f"- Description: {description}\n"
        
        # Add similar artworks context if available
        similar_context = ""
        if similar_artworks:
            similar_context = "\nSimilar artworks in the collection:\n"
            for i, similar in enumerate(similar_artworks, 1):
                similar_context += f"{i}. {similar.get('artist', 'Unknown')} - {similar.get('genre', 'Unknown')} ({similar.get('style', 'Unknown')})\n"
        
        prompt = f"""You are an art historian and curator. Create a concise, engaging backstory for an artwork based on the following information.

{artwork_context}{similar_context}

Write a backstory (2-4 sentences, maximum 200 words) that:
1. Provides context about the artwork's artistic significance
2. Mentions the artist's style and the genre
3. Highlights what makes this artwork notable or interesting
4. Uses engaging, accessible language suitable for art enthusiasts

Be factual but engaging. If specific historical details are not available, focus on the artistic style, genre characteristics, and what makes this type of artwork meaningful.

Backstory:"""
        
        return prompt
    
    def generate_backstory(
        self,
        artwork_id: str,
        use_similar_artworks: bool = True,
        model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """
        Generate a backstory for an artwork.
        
        Args:
            artwork_id: ID of the artwork (e.g., "shard1_0" or numeric index)
            use_similar_artworks: Whether to use similar artworks for context
            model: OpenAI model to use (default: gpt-4o-mini)
        
        Returns:
            Dictionary with 'backstory' and 'metadata' keys
        """
        # Get artwork metadata
        metadata = self._get_artwork_metadata(artwork_id)
        if not metadata:
            raise ValueError(f"Artwork with ID {artwork_id} not found")
        
        # Find similar artworks if requested
        similar_artworks = []
        if use_similar_artworks:
            similar_artworks = self._find_similar_artworks(artwork_id, top_k=3)
        
        # Build prompt
        prompt = self._build_prompt(metadata, similar_artworks)
        
        # Call OpenAI API
        try:
            response = self.client_openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert art historian and curator who writes engaging, concise backstories for artworks."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            backstory = response.choices[0].message.content.strip()
            
            return {
                "backstory": backstory,
                "artwork_id": artwork_id,
                "artist": metadata.get("artist_name", "Unknown"),
                "genre": metadata.get("genre_name", "Unknown"),
                "style": metadata.get("style_name", "Unknown"),
                "similar_artworks_used": len(similar_artworks)
            }
        except Exception as e:
            raise RuntimeError(f"Error generating backstory: {str(e)}")


def generate_backstory_for_artwork(
    artwork_id: str,
    chroma_db_path: str,
    metadata_json_path: str,
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate backstory for a single artwork.
    
    Args:
        artwork_id: ID of the artwork
        chroma_db_path: Path to ChromaDB directory
        metadata_json_path: Path to wiki_art_data.json
        openai_api_key: Optional OpenAI API key
    
    Returns:
        Dictionary with backstory and metadata
    """
    generator = BackstoryGenerator(
        chroma_db_path=chroma_db_path,
        metadata_json_path=metadata_json_path,
        openai_api_key=openai_api_key
    )
    return generator.generate_backstory(artwork_id)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_backstory.py <artwork_id>")
        sys.exit(1)
    
    artwork_id = sys.argv[1]
    
    DATA_DIR = "/Users/srikala/projects/AI-Portfolio/art_identifier/data"
    CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
    METADATA_JSON = os.path.join(DATA_DIR, "json", "wiki_art_data.json")
    
    try:
        result = generate_backstory_for_artwork(
            artwork_id=artwork_id,
            chroma_db_path=CHROMA_DB_DIR,
            metadata_json_path=METADATA_JSON
        )
        print("\n" + "="*60)
        print("BACKSTORY")
        print("="*60)
        print(result["backstory"])
        print("\n" + "="*60)
        print("METADATA")
        print("="*60)
        print(f"Artist: {result['artist']}")
        print(f"Genre: {result['genre']}")
        print(f"Style: {result['style']}")
        print(f"Similar artworks used: {result['similar_artworks_used']}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

