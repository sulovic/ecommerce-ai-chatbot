# retriever.py
import json
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "context_index.faiss"
METADATA_FILE = "context_metadata.json"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 1

# Load embedding model and FAISS index once
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

def retrieve_context(question: str) -> str:
    # Prefix "query:" as required by e5 model
    query = "query: " + question
    query_vec = embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
    scores, indices = index.search(query_vec, TOP_K)
    if len(indices[0]) > 0 and indices[0][0] < len(metadata):
        return metadata[indices[0][0]]["text"]
    return ""
