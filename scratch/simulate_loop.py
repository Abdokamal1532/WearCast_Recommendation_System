# Simulation Script: Verifying Cecile's Recommendations
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.getcwd())
import config
from src.models.content_based import ContentBasedModel
from src.models.collaborative import CollaborativeModel
from src.models.hybrid import HybridModel
from src.preprocessor import load_processed

def demo():
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = "09afee19-8a92-4b58-8531-634db48fd3f2"
    
    print(f"\n--- SIMULATION START: User {user_id} ---")
    
    # 1. Load data
    interactions, item_features, user_profiles = load_processed()
    with open('data/raw/events.json', 'r') as f:
        raw_data = json.load(f)
    catalog = {p['productId']: p for p in raw_data['catalog']}
    
    # 2. Setup Hybrid Model
    cb = ContentBasedModel(item_features, user_profiles)
    cf = CollaborativeModel(interactions) 
    hybrid = HybridModel(cb, cf, interactions, list(catalog.values()))
    hybrid_model = hybrid # For naming consistency
    
    # 3. Get Recommendations
    recs = hybrid.recommend(user_id, top_k=50)
    
    print("\n[AI OUTPUT] Top 5 Recommended Categories for this User Profile:")
    print("-" * 60)
    for i, r in enumerate(recs):
        pid = r['productId']
        product_info = catalog.get(pid, {"categoryName": "Unknown", "price": 0})
        print(f"{i+1}. [{pid}]")
        print(f"   Category: {product_info['categoryName']} | Style: {product_info.get('dressStyle', 'N/A')} | Price: ${product_info['price']:.2f}")
    print("-" * 60)
    
    # Check if the intent (Dresses) is honored
    top_cat = catalog.get(recs[0]['productId'], {}).get('categoryName')
    if top_cat == "Dresses":
        print("\n[VERIFICATION] SUCCESS! The AI correctly reacted to the 'Dresses' filter.")
    else:
        print(f"\n[VERIFICATION] Result: {top_cat}")

if __name__ == "__main__":
    demo()
