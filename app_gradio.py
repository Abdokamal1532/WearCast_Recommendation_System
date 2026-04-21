# =============================================================================
# app_gradio.py
# -----------------------------------------------------------------------------
# Web UI for the Recommendation System using Gradio.
# Generates a Public URL using `share=True`.
# =============================================================================

import os
import json
import gradio as gr
from src.preprocessor import load_processed
from src.models.content_based import ContentBasedModel
from src.models.collaborative import CollaborativeModel
from src.models.hybrid import HybridModel
from src.retrieval import CandidateRetriever
from src.ranking import Ranker
import config

print("[Gradio] Loading System Models...")
interactions_df, item_features_df, user_profiles_df = load_processed()

# Load Original Catalog
with open(config.RAW_EVENTS_PATH, "r", encoding="utf-8") as f:
    catalog = json.load(f)["catalog"]

# Initialize Models
cb_model = ContentBasedModel(item_features_df, user_profiles_df)
cf_model = CollaborativeModel.load(interactions_df)
hybrid_model = HybridModel(cb_model, cf_model, interactions_df)
retriever = CandidateRetriever(cb_model, interactions_df)
ranker = Ranker(hybrid_model, item_features_df, catalog)

# Get sample users to select from
sample_users = interactions_df["userId"].unique().tolist()[:50] 
category_choices = ["All"] + config.CATEGORIES

def get_recommendations(user_id, category, min_price, max_price, top_k):
    # Prepare Filters
    filters = {
        "minPrice": min_price,
        "maxPrice": max_price,
    }
    if category != "All":
        filters["categoryName"] = category
        
    candidates = retriever.retrieve(user_id=user_id, exclude_seen=True)
    
    if not candidates:
        ranked = hybrid_model.trending_fallback(top_k)
    else:
        ranked = ranker.rank(
            user_id=user_id,
            candidates=candidates,
            top_k=top_k,
            filters=filters,
        )

    # Format the Output as a clean HTML table or Markdown
    output_md = "### Top Recommendations Context\n\n"
    output_md += "| Rank | Product ID | Category | Style | Price | Match Score |\n"
    output_md += "|------|------------|----------|-------|-------|-------------|\n"
    
    for i, item in enumerate(ranked, 1):
        score = item.get("score", 0.0)
        # Convert float to percentage
        pct = round(score * 100, 1)
        price = item.get("price", 0)
        cat = item.get("categoryName", "N/A")
        style = item.get("dressStyle", "N/A")
        pid = item.get("productId", "")
        
        output_md += f"| **{i}** | `{pid}` | {cat} | {style} | ${price} | **{pct}%** |\n"
        
    if not ranked:
        output_md = "❌ No items found passing these strict filters. Try broadening the price or category."

    return output_md

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🛍️ AI Recommendation System Engine")
    gr.Markdown("Select a user and tweak the filters to see how the mathematical Hybrid Engine adapts live.")
    
    with gr.Row():
        with gr.Column(scale=1):
            user_dropdown = gr.Dropdown(choices=sample_users, value=sample_users[0], label="1. Select User Context")
            cat_dropdown = gr.Dropdown(choices=category_choices, value="All", label="2. Desired Category")
            
            with gr.Row():
                min_price_slider = gr.Slider(0, 2000, value=0, label="Min Price ($)", step=10)
                max_price_slider = gr.Slider(0, 2000, value=2000, label="Max Price ($)", step=10)
                
            top_k_slider = gr.Slider(1, 20, value=10, label="Top K Items", step=1)
            
            submit_btn = gr.Button("🔍 Fetch Recommendations", variant="primary")
            
        with gr.Column(scale=2):
            output_display = gr.Markdown("Waiting for input...")

    submit_btn.click(
        fn=get_recommendations,
        inputs=[user_dropdown, cat_dropdown, min_price_slider, max_price_slider, top_k_slider],
        outputs=output_display
    )

if __name__ == "__main__":
    print("[Gradio] Launching Public Server...")
    # 🌟 share=True is what provides your public URL!
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
