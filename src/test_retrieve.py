import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="resume_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
collection = client.get_collection(name="resume_bullets", embedding_function=ef)

query = "led a team of students to organize a college event"
results = collection.query(query_texts=[query], n_results=5)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[{meta['source']}] {doc}\n")