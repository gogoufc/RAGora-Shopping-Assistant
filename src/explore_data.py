import pickle
import random
import os
from collections import Counter

def explore_data():
    metadata_path = os.path.join("..", "data", "c4_metadata.pkl")
    
    if not os.path.exists(metadata_path):
        print("Metadata file not found!")
        return

    # Load the dataset
    print("Loading data...")
    with open(metadata_path, "rb") as f:
        item_pool = pickle.load(f)

    # Print total number of products
    print(f"Total products: {len(item_pool)}\n")
    
    # Extract categories and count their frequencies
    categories = [item.get("category", "Unknown") for item in item_pool]
    cat_counts = Counter(categories)

    # Print the top 20 most frequent categories
    print("--- Top 20 Categories in the Dataset ---")
    for cat, count in cat_counts.most_common(20):
        print(f"{cat}: {count} products")

    # Print the exact structure of the first item to show the schema
    print("\n=== Structure of the first item in the dataset ===")
    print(item_pool[0])

    # Dynamically get the names of the top 3 categories
    top_3_categories = [cat for cat, count in cat_counts.most_common(3)]
    
    # Loop through the top 3 categories and pick 3 random items for each
    for cat in top_3_categories:
        print(f"\n=== 3 Random Products in category: {cat} ===")
        
        # Filter items belonging to the current category
        cat_items = [item for item in item_pool if item.get("category") == cat]
        
        # Select up to 3 random items safely
        sample_items = random.sample(cat_items, min(3, len(cat_items)))
        
        for item in sample_items:
            # Print the item dictionary as a string, truncated to 150 characters
            print(f"- {str(item)[:150]}...\n")

if __name__ == "__main__":
    explore_data()