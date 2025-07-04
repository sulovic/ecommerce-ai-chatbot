import json
import faiss
from sentence_transformers import SentenceTransformer
import os

INDEX_FILE = "./vector_store/context_index.faiss"
METADATA_FILE = "./vector_store/context_metadata.json"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 1

# Ensure index file exists and metadata file exists
if not os.path.exists(INDEX_FILE):
    raise FileNotFoundError(f"Index file not found: {INDEX_FILE}")
if not os.path.exists(METADATA_FILE):   
    raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

# Load embedding model and FAISS index once
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Loaded embedding model, FAISS index and metatdata.")

def retrieve_context(question: str) -> str:
    # Prefix "query:" as required by e5 model
    query = "query: " + question
    query_vec = embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
    scores, indices = index.search(query_vec, TOP_K)
    if indices.size == 0 or indices[0][0] >= len(metadata):
        print("No relevant context found.")
        return ""

    match_idx = indices[0][0]
    context = metadata[match_idx]
    return context

if __name__ == "__main__":
    # Test usage
    test_question = "Koje su dimenzije KKB90176?"
    context = retrieve_context(test_question)
    print(f"Context for '{test_question}': {context}")
