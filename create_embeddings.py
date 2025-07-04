import pandas as pd
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup


# Files & constants
PRODUCT_CSV = 'products_data.csv'
QA_JSON = 'serbian_qa.json'
MODEL_NAME = "intfloat/multilingual-e5-small"
INDEX_FILE = 'context_index.faiss'
METADATA_FILE = 'context_metadata.json'

# Load product data
print("Loading products from CSV...")
df = pd.read_csv(PRODUCT_CSV, encoding='utf-8')

product_texts = []
product_metadata = []

for _, row in df.iterrows():
    sku = row['sku']
    name = row['name']

    # Convert description HTML to plain text safely
    raw_description = row.get('description', '') or ''
    soup = BeautifulSoup(raw_description, 'html.parser')
    description = soup.get_text(separator=' ', strip=True)    
    
    url_key = row.get('url_key', '') or ''
    base_image = row.get('base_image', '') or ''

    text = f"product: SKU: {sku}\nName: {name}\nDescription: {description}\nURL: {url_key}"
    product_texts.append(text)
    product_metadata.append({
        "type": "product",
        "sku": sku,
        "name": name,
        "description": description,
        "url_key": url_key,
        "base_image": base_image
    })

# Load QA data
print("Loading QA data from JSON...")
with open(QA_JSON, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

qa_texts = []
qa_metadata = []

for item in qa_data:
    question = item.get('question', '')
    answer = item.get('answer', '')
    text = f"qa: question: {question} answer: {answer}"
    qa_texts.append(text)
    qa_metadata.append({
        "type": "qa",
        "question": question,
        "answer": answer
    })

# Combine all texts and metadata
print("Combining product and QA data...")
all_texts = product_texts + qa_texts
all_metadata = product_metadata + qa_metadata

# Load embedding model
device = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
print(f"Loading embedding model on device: {device}")
model = SentenceTransformer(MODEL_NAME, device=device)

print(f"Encoding {len(all_texts)} combined entries...")
embeddings = model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)

# Build FAISS index
dimension = embeddings.shape[1]
print(f"Building FAISS index with dimension {dimension}...")
index = faiss.IndexFlatIP(dimension) 
index.add(embeddings)

# Save index and metadata
print(f"Saving FAISS index to {INDEX_FILE}...")
faiss.write_index(index, INDEX_FILE)

print(f"Saving combined metadata to {METADATA_FILE}...")
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

print("Done.")
