import chromadb, numpy as np
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("artworks")
embs = collection.get(limit=5, include=["embeddings"])["embeddings"]
for i, e in enumerate(embs):
    print(f"Item {i} norm:", np.linalg.norm(e))