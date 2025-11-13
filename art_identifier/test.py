#Option1
import chromadb
client = chromadb.PersistentClient(path="art_identifier/data/chroma_db")
collection = client.get_collection("wikiart")
print(collection.count())

#Option2
from chromadb import PersistentClient

# Path to your ChromaDB directory
chroma_db_path = "/Users/srikala/projects/AI-Portfolio/art_identifier/data/chroma_db"

# Connect to ChromaDB
client = PersistentClient(path=chroma_db_path)

# List all collections
collections = client.list_collections()
print("Collections in this ChromaDB:")
for c in collections:
    print(" -", c.name)

collection = client.get_collection("wikiart")
print(len(collection.get()['ids'])) 
