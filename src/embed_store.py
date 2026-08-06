import json
import chromadb
from chromadb.utils import embedding_functions

INPUT_FILE = "data/processed/chunks_clean.json"
DB_PATH = "resume_db"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks to embed")

client = chromadb.PersistentClient(path=DB_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")

collection = client.get_or_create_collection(
    name="resume_bullets",
    embedding_function=ef
)

# Batch in chunks of 500 to avoid overloading memory
BATCH_SIZE = 500
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    collection.add(
        documents=[c["text"] for c in batch],
        metadatas=[{"source": c["source"]} for c in batch],
        ids=[f"chunk_{i+j}" for j in range(len(batch))]
    )
    print(f"Embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

print("Done. Collection count:", collection.count())