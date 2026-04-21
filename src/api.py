# =============================================================================
# src/api.py
# -----------------------------------------------------------------------------
# FastAPI REST endpoint to serve recommendations.
#
# Start the server:
#   uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
#
# Endpoints:
#   GET  /health               — liveness check
#   POST /recommend            — get top-K recommendations for a user
#   GET  /trending             — get globally trending products
# =============================================================================

import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.preprocessor import load_processed
from src.models.content_based import ContentBasedModel
from src.models.collaborative import CollaborativeModel
from src.models.hybrid import HybridModel
from src.retrieval import CandidateRetriever
from src.ranking import Ranker

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------

_state: dict = {}


def _load_catalog() -> list[dict]:
    try:
        with open(config.RAW_EVENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("catalog", [])
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"[API] [WARN] Catalog file not found or invalid: {config.RAW_EVENTS_PATH}")
        return []


def initialize_engine():
    """Logic to load/reload all models and data into the global state."""
    print("\n[API] [SYSTEM] Initializing AI Engine...")
    try:
        # Step 1: Force a sync with the Database
        try:
            from src.db_loader import sync
            print("[API] [DB] Synchronizing with SQL Server on startup...")
            sync()
        except Exception as sync_err:
            print(f"[API] [DB] [WARN] Synchronization failed: {sync_err}")

        # Step 2: Load data
        interactions_df, item_features_df, user_profiles_df = load_processed()
        catalog = _load_catalog()

        # Step 3: Initialize models
        cb_model = ContentBasedModel(item_features_df, user_profiles_df)
        cf_model = CollaborativeModel.load(interactions_df)
        hybrid_model = HybridModel(cb_model, cf_model, interactions_df, catalog)
        retriever = CandidateRetriever(cb_model, interactions_df)
        ranker = Ranker(hybrid_model, item_features_df, catalog)

        _state["cb"] = cb_model
        _state["cf"] = cf_model
        _state["hybrid"] = hybrid_model
        _state["retriever"] = retriever
        _state["ranker"] = ranker
        _state["interactions_df"] = interactions_df
        _state["catalog"] = catalog

        print("[API] [SUCCESS] AI Engine is now online.")
    except Exception as e:
        print(f"[API] [CRITICAL ERROR] Failed to initialize AI Engine: {e}")
        # We don't raise 'e' here to prevent the FastAPI process from crashing on startup.
        # Endpoints will handle cases where the state is missing.


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models in the background to prevent Vercel startup timeouts (10s limit).
    The app will start immediately, and models will load shortly after.
    """
    import asyncio
    # Start initialization in the background
    asyncio.create_task(asyncio.to_thread(initialize_engine))
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Recommendation System API",
    description="Hybrid CF + CB recommendation engine for e-commerce events.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class FilterContext(BaseModel):
    minPrice: Optional[float] = Field(None, description="Minimum product price")
    maxPrice: Optional[float] = Field(None, description="Maximum product price")
    targetAudience: Optional[list[str]] = Field(None, description="e.g. ['WOMEN', 'MEN']")
    dressStyle: Optional[str] = Field(None, description="e.g. FORMAL, CASUAL")
    categoryName: Optional[str] = Field(None, description="e.g. Dresses")
    sellerId: Optional[str] = Field(None, description="e.g. seller_001")


class RecommendRequest(BaseModel):
    userId: str = Field(..., description="ID of the user to get recommendations for")
    topK: int = Field(config.API_DEFAULT_TOP_K, ge=1, le=50, description="Number of recommendations")
    filters: Optional[FilterContext] = Field(None, description="Optional filter constraints")
    excludeSeen: bool = Field(False, description="Exclude already-interacted products")


class RecommendedItem(BaseModel):
    productId: str
    score: float
    cb_score: float
    cf_score: float
    price: Optional[float]
    categoryName: Optional[str]
    dressStyle: Optional[str]
    targetAudience: Optional[object]
    sellerId: Optional[str]
    productType: Optional[str]


class RecommendResponse(BaseModel):
    userId: str
    topK: int
    model: str = "hybrid"
    recommendations: list[RecommendedItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Liveness check."""
    return {"status": "ok", "service": "recommendation-system"}


@app.post("/internal/reload", tags=["System"])
def reload_models():
    """
    Triggers a hot-reload of the AI models.
    Called by the background scheduler after training finishes.
    """
    print("\n[API] [SIGNAL] Received Hot-Reload request from internal scheduler.")
    try:
        initialize_engine()
        return {"status": "reloaded", "timestamp": str(datetime.now())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(request: RecommendRequest):
    """
    Get personalised product recommendations for a user.

    - **userId**: the user to recommend for
    - **topK**: number of results (default 10)
    - **filters**: optional hard constraints (price, style, category, etc.)
    - **excludeSeen**: whether to remove already-interacted products
    """
    print(f"\n[API] [REQUEST] POST /recommend | User: {request.userId} | topK: {request.topK} | excludeSeen: {request.excludeSeen}")
    retriever: CandidateRetriever = _state.get("retriever")
    ranker: Ranker = _state.get("ranker")

    if retriever is None or ranker is None:
        print("[API] [SERVICE UNAVAILABLE] Models not loaded yet.")
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    # Stage 1 — Retrieval
    print(f"[API] [STAGE 1] Starting retrieval for {request.userId}...")
    try:
        candidates = retriever.retrieve(
            user_id=request.userId,
            exclude_seen=request.excludeSeen,
        )
        print(f"[API] [OK] Retrieval complete. Found {len(candidates)} candidates.")
    except Exception as e:
        print(f"[API] [ERROR] Retrieval failure: {e}")
        candidates = []

    if not candidates:
        # Cold-start or error: return trending
        print(f"[API] [FALLBACK] No candidates. Returning trending items for user {request.userId}")
        trending = _state["hybrid"].trending_fallback(request.topK)
        return RecommendResponse(
            userId=request.userId,
            topK=request.topK,
            model="trending (fallback)",
            recommendations=[RecommendedItem(**item) for item in trending],
        )

    # Stage 2 — Ranking
    print(f"[API] [STAGE 2] Starting ranking for {len(candidates)} candidates...")
    try:
        # Check if ranking filters are provided
        filter_dict = request.filters.model_dump(exclude_none=True) if request.filters else None
        
        ranked = ranker.rank(
            user_id=request.userId,
            candidates=candidates,
            top_k=request.topK,
            filters=filter_dict
        )
        print(f"[API] [SUCCESS] Ranking complete. Returning top {len(ranked)} items for {request.userId}")
        return RecommendResponse(
            userId=request.userId,
            topK=request.topK,
            recommendations=[RecommendedItem(**item) for item in ranked],
        )
    except Exception as e:
        print(f"[API] [ERROR] Ranking failure: {e}")
        # Fallback to unranked retrieval
        print(f"[API] [FALLBACK] Returning unranked retrieval candidates due to ranking error.")
        fallback_recs = [{"productId": pid, "score": 0.0, "cb_score": 0.0, "cf_score": 0.0} for pid in candidates[:request.topK]]
        return RecommendResponse(
            userId=request.userId,
            topK=request.topK,
            model="retrieval (fallback)",
            recommendations=[RecommendedItem(**item) for item in fallback_recs],
        )


@app.get("/trending", tags=["Recommendations"])
def trending(top_k: int = 10):
    """Return globally trending products (by total interaction weight)."""
    hybrid: HybridModel = _state.get("hybrid")
    if hybrid is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    catalog_map = {item["productId"]: item for item in _state["catalog"]}
    results = []
    for item in hybrid.trending_fallback(top_k):
        pid = item["productId"]
        prod = catalog_map.get(pid, {})
        results.append({
            "productId": pid,
            "categoryName": prod.get("categoryName"),
            "price": prod.get("price"),
            "dressStyle": prod.get("dressStyle"),
            "targetAudience": prod.get("targetAudience"),
        })
    return {"trending": results}
