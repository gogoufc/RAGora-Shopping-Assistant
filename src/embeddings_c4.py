# embeddings_c4.py
import json
import os
import pickle
import faiss
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# 1. Setup Paths (Create data directory if it doesn't exist)
# NOTE: If you run this from the 'src' folder, use "../data".
# If you run from the root folder, change this to "data".
DATA_DIR = "../data" 

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

INDEX_PATH = os.path.join(DATA_DIR, "faiss_index_c4.bin")
METADATA_PATH = os.path.join(DATA_DIR, "c4_metadata.pkl")

print("Downloading dataset from Hugging Face...")
filepath = hf_hub_download(
    repo_id='McAuley-Lab/Amazon-C4',
    filename='sampled_item_metadata_1M.jsonl',
    repo_type='dataset'
)

# 2. Load Products & Prepare Text for Embeddings
item_pool = []
texts = []

print("Reading dataset and preparing text...")
with open(filepath, 'r') as f:
    for line in f:
        item = json.loads(line)
        
        # Keep the raw object for metadata storage
        item_pool.append(item)
        
        # CONSTRUCT TEXT FOR EMBEDDING:
        # The Amazon-C4 dataset collapses the title, description, and brand features into a 
        #single unstructured 'metadata' field.
        # Using .get() prevents errors if keys are missing.
        metadata = item.get('metadata', '')
        category = item.get('category', '')
        
        # Concatenate the category with the metadata.
        full_text = f"{category} {metadata}".strip()
        texts.append(full_text)

print(f"Loaded {len(item_pool)} products.")

# 3. Create Embeddings
print("Generating embeddings (this will take time)...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
# Batch size is set to 32 to be safe on memory.
embeddings = model.encode(
    texts, 
    batch_size=32, 
    show_progress_bar=True, 
    convert_to_numpy=True
)

# 4. Create & Save FAISS Index
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"Saving to {DATA_DIR}...")
faiss.write_index(index, INDEX_PATH)

with open(METADATA_PATH, "wb") as f:
    pickle.dump(item_pool, f)

print("Success! FAISS index and metadata saved.")