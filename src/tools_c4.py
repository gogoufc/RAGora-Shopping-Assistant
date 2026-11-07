# tools_c4.py
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

# 1. SETUP PATHS
# Determine the absolute path to the 'data' directory (one level up from 'src')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "data")

# Define file paths
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index_c4.bin")
METADATA_FILE = os.path.join(DATA_DIR, "c4_metadata.pkl")

# Global variables to hold loaded resources
index = None
item_pool = []
model = None

# 2. LOAD RESOURCES
try:
    print(f"Checking for data in: {DATA_DIR}")
    
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        print("Loading FAISS index...")
        index = faiss.read_index(INDEX_FILE)
        
        print("Loading Metadata...")
        with open(METADATA_FILE, "rb") as f:
            item_pool = pickle.load(f)
            
        print("Loading Embedding Model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("Search tools loaded successfully.")
    else:
        print(f"ERROR: Files not found in {DATA_DIR}")
        print("   Please run 'python embeddings_c4.py' first.")

except Exception as e:
    print(f"Error loading resources: {e}")

def search_products(query: str, top_k: int = 5):
    """
    Search for products semantically using FAISS.
    Returns a list of dictionaries with product details.
    """
    # Safety check to ensure the system is initialized
    if index is None or model is None:
        return [{"title": "Error: System not initialized", "description": "Please check logs."}]

    try:
        # Encode the user query into a vector
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Perform the search in the FAISS index (retrieve nearest neighbors)
        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        # Iterate over the found indices
        for i, idx in enumerate(indices[0]):
            # Ensure the index is within bounds of our dataset
            if idx < len(item_pool) and idx >= 0:
                item = item_pool[idx]
                
                # EXTRACTION LOGIC FOR C4 DATASET
                # Since the dataset only has 'metadata' (a big text blob), 
                # we need to parse it manually.
                
                full_text = item.get("metadata", "")
                
                # Heuristic: The Title is usually the text before the first period (.)
                # or the first 100 characters if no period is found.
                if "." in full_text:
                    title_candidate = full_text.split(".")[0]
                else:
                    title_candidate = full_text[:100]
                
                # Clean up title if it's too long
                if len(title_candidate) > 100:
                    title = title_candidate[:100] + "..."
                else:
                    title = title_candidate
                
                # The description is the rest of the text
                description = full_text.replace(title_candidate, "").strip()
                # Truncate description to keep it manageable
                formatted_desc = f"{description[:250]}..." if len(description) > 250 else description
                
                # Note: This version of the C4 dataset does not contain price/rating fields.
                price = "N/A" 
                
                results.append({
                    "title": title,
                    "price": price, 
                    "description": formatted_desc,
                    "category": item.get("category", "Unknown"),
                    "item_id": item.get("item_id", "N/A"),
                    "score": float(distances[0][i]) # L2 Distance
                })
        
        return results

    except Exception as e:
        print(f"Error during search: {e}")
        return []


#if item_pool:
    #sample = item_pool[509]
    #full_text = sample.get("metadata", "")
    #print("RAW:", full_text[:200])
    #title = full_text.split(".")[0] if "." in full_text else full_text[:100]
    #description = full_text.replace(title, "").strip()[:250]
    #print("TITLE:", title)
    #print("DESC:", description)
    
    
if __name__ == "__main__":
    # Test block: Runs only if this file is executed directly
    if index:
        print("\n--- TEST SEARCH: 'gaming laptop' ---")
        results = search_products("gaming laptop under 1000", top_k=3)
        for r in results:
            print(f"Found: {r['title']}")
            print(f"Price: {r['price']} | Score: {r['score']:.4f}")
            print("-" * 30)