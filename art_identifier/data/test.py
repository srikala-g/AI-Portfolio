from chromadb import PersistentClient

# Path to your ChromaDB directory
chroma_db_path = "/Users/srikala/projects/AI-Portfolio/art_identifier/data/Chroma_db"

# Connect to ChromaDB
client = PersistentClient(path=chroma_db_path)

# List all collections
collections = client.list_collections()
print("Collections in this ChromaDB:")
for c in collections:
    print(" -", c.name)

collection = client.get_collection("wikiart")
print(len(collection.get()['ids'])) 



# print("Total IDs:", len(collection.get()['ids']))
# print("Sample metadata:", collection.get()['metadatas'][0][:3])