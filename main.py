# =============================================================================
# main.py — Consolidated Bulletproof Vercel Entrypoint
# =============================================================================

import os
import sys
import json
from contextlib import asynccontextmanager
from typing import Optional, Any
from datetime import datetime

# Add root to sys.path for local running support
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "status": "initializing",
    "error": None
}

# ---------------------------------------------------------------------------
# Lazy-loading Engine
# ---------------------------------------------------------------------------

def initialize_engine():
    """Logic to load/reload all models with lazy imports for Vercel speed."""
    global _state
    print("\n[SYSTEM] Starting AI Engine initialization (Lazy Load)...")
    
    try:
        # Move heavy imports INSIDE to prevent startup timeout
        import pandas as pd
        import numpy as np
        import config
        from src.preprocessor import load_processed
        from src.models.content_based import ContentBasedModel
        from src.models.collaborative import CollaborativeModel
        from src.models.hybrid import HybridModel
        from src.retrieval import CandidateRetriever
        from src.ranking import Ranker

        # Step 1: Sync (optional/resilient)
        try:
            from src.db_loader import sync
            sync()
        except Exception as e:
            print(f"[DB] [WARN] Sync skipped/failed: {e}")

        # Step 2: Load data
        interactions_df, item_features_df, user_profiles_df = load_processed()
        
        # Load catalog
        catalog = []
        if os.path.exists(config.RAW_EVENTS_PATH):
            with open(config.RAW_EVENTS_PATH, "r", encoding="utf-8") as f:
                catalog = json.load(f).get("catalog", [])

        # Step 3: Initialize models
        cb_model = ContentBasedModel(item_features_df, user_profiles_df)
        cf_model = CollaborativeModel.load(interactions_df)
        hybrid_model = HybridModel(cb_model, cf_model, interactions_df, catalog)
        retriever = CandidateRetriever(cb_model, interactions_df)
        ranker = Ranker(hybrid_model, item_features_df, catalog)

        # Update state
        _state.update({
            "cb": cb_model,
            "cf": cf_model,
            "hybrid": hybrid_model,
            "retriever": retriever,
            "ranker": ranker,
            "interactions_df": interactions_df,
            "catalog": catalog,
            "status": "online",
            "last_updated": str(datetime.now())
        })
        print("[SUCCESS] AI Engine is now online.")

    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        print(f"[CRITICAL] {error_msg}")
        _state["status"] = "error"
        _state["error"] = error_msg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up logic: fire and forget initialization to keep Vercel happy."""
    print("[SYSTEM] FastAPI Lifespan starting...")
    import asyncio
    
    # Define a helper to run the sync function in a thread safely
    def run_init():
        initialize_engine()

    # Use the running loop to schedule the initialization
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, run_init)
    
    yield
    print("[SYSTEM] FastAPI Lifespan shutting down.")
    _state.clear()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WearCast Recommender",
    version="2.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FilterContext(BaseModel):
    minPrice: Optional[float] = None
    maxPrice: Optional[float] = None
    targetAudience: Optional[list[str]] = None
    dressStyle: Optional[str] = None
    categoryName: Optional[str] = None
    sellerId: Optional[str] = None

class RecommendRequest(BaseModel):
    userId: str
    topK: int = 10
    filters: Optional[FilterContext] = None
    excludeSeen: bool = False

class RecommendedItem(BaseModel):
    productId: str
    score: float
    cb_score: float
    cf_score: float
    price: Optional[float] = None
    categoryName: Optional[str] = None
    dressStyle: Optional[str] = None
    targetAudience: Optional[Any] = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/health")
def health():
    return {
        "status": _state.get("status"),
        "engine": "active",
        "error": _state.get("error"),
        "last_updated": _state.get("last_updated")
    }

@app.post("/recommend")
def recommend(request: RecommendRequest):
    retriever = _state.get("retriever")
    ranker = _state.get("ranker")
    
    if not retriever or not ranker:
        raise HTTPException(status_code=503, detail="AI Engine is still loading or failed. Check /health.")

    try:
        candidates = retriever.retrieve(request.userId, exclude_seen=request.excludeSeen)
        
        # Fallback to trending if no candidates
        if not candidates:
            trending = _state["hybrid"].trending_fallback(request.topK)
            return {"userId": request.userId, "model": "trending (fallback)", "recommendations": trending}

        filter_dict = request.filters.model_dump(exclude_none=True) if request.filters else None
        ranked = ranker.rank(request.userId, candidates, top_k=request.topK, filters=filter_dict)
        
        return {"userId": request.userId, "recommendations": ranked}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trending")
def trending(top_k: int = 10):
    hybrid = _state.get("hybrid")
    if not hybrid:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    return {"trending": hybrid.trending_fallback(top_k)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
