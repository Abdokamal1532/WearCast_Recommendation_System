# =============================================================================
# src/models/hybrid.py
# -----------------------------------------------------------------------------
# Hybrid Recommender: blends Content-Based (CB) and Collaborative Filtering (CF)
# scores using a configurable alpha weight.
#
#   final_score = α × CF_score + (1 − α) × CB_score
#
# Falls back gracefully:
#   • Unknown user (cold-start) → CB only (or trending if no CB profile)
#   • Item not in CF model     → CB score only
# =============================================================================

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.models.content_based import ContentBasedModel
from src.models.collaborative import CollaborativeModel


class HybridModel:
    """
    Weighted linear combination of CF and CB scores.

    Parameters
    ----------
    cb_model : ContentBasedModel
    cf_model : CollaborativeModel
    interactions_df : pd.DataFrame
        Used to determine already-seen items and build trending fallback.
    alpha : float
        Weight for CF score (0 = pure CB, 1 = pure CF). Default from config.
    """

    def __init__(
        self,
        cb_model: ContentBasedModel,
        cf_model: CollaborativeModel,
        interactions_df: pd.DataFrame,
        catalog: list[dict],
        alpha: float = config.HYBRID_ALPHA,
    ):
        self.cb_model = cb_model
        self.cf_model = cf_model
        self.interactions_df = interactions_df
        self.alpha = alpha
        
        # Build product metadata map for quick lookups (CreatedOn)
        self.catalog_meta = {p["productId"]: p for p in catalog}

        # Pre-build trending list (top-N most interacted products globally)
        self._trending: list[str] = (
            interactions_df.groupby("productId")["weight"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

        # Map userId → set of already-purchased product ids
        self._user_interactions: dict[str, set[str]] = (
            interactions_df.groupby("userId")["productId"]
            .apply(set)
            .to_dict()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _min_max_normalize(self, score_dict: dict[str, float]) -> dict[str, float]:
        """Scale scores to [0, 1] range."""
        if not score_dict:
            return {}
        values = list(score_dict.values())
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return {k: 1.0 for k in score_dict}
        return {k: (v - vmin) / (vmax - vmin) for k, v in score_dict.items()}

    def _get_seen_items(self, user_id: str) -> set[str]:
        return self._user_interactions.get(user_id, set())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        top_k: int = config.RANKING_TOP_K,
        candidate_pool: list[str] | None = None,
    ) -> list[dict]:
        """
        Return top-K recommendations for *user_id*.

        Parameters
        ----------
        user_id : str
        top_k : int
        candidate_pool : list[str] | None
            Pre-filtered candidate product IDs (from retrieval stage).
            If None, all known products are candidates.

        Returns
        -------
        list of dicts: [{"productId": str, "score": float, "cb_score": float, "cf_score": float}]
        """
        seen = self._get_seen_items(user_id)

        # Determine candidate set
        if candidate_pool is not None:
            candidates = [p for p in candidate_pool if p not in seen]
        else:
            candidates = [p for p in self.cf_model.get_all_product_ids() if p not in seen]

        if not candidates:
            # Fallback to trending
            candidates = [p for p in self._trending if p not in seen][:top_k * 2]

        # --- CB scores ---
        cb_raw = self.cb_model.score_items(user_id, candidates)
        cb_norm = self._min_max_normalize(cb_raw)

        # --- CF scores ---
        # Safeguard: if CF model is not trained, skip scoring it
        if self.cf_model.model is not None:
            cf_raw = self.cf_model.score_items(user_id, candidates)
            cf_norm = self._min_max_normalize(cf_raw)
        else:
            cf_norm = {}

        # --- Blend ---
        blended = []
        for pid in candidates:
            cb_s = cb_norm.get(pid, 0.0)
            cf_s = cf_norm.get(pid, 0.0)

            # If user is cold-start in CF, weight shifts to CB entirely
            is_cold_start_cf = (user_id not in self.cf_model._user_id_map)
            effective_alpha = 0.0 if is_cold_start_cf else self.alpha

            score = effective_alpha * cf_s + (1.0 - effective_alpha) * cb_s
            
            # --- Freshness/Discovery Boost ---
            # Boost items created recently (proxy for 'New Arrivals')
            boost = 1.0
            p_meta = self.catalog_meta.get(pid, {})
            c_date_str = p_meta.get("createdOn")
            if c_date_str:
                try:
                    c_date = pd.to_datetime(c_date_str)
                    age_days = (pd.Timestamp.now() - c_date).days
                    if age_days <= 1: # Product is fresh!
                        boost = 1.25 if is_cold_start_cf else 1.10
                except: pass
            
            final_score = score * boost

            # Log first 3 candidates for transparency
            if len(blended) < 3:
                print(f"[Hybrid] [DEBUG] User: {user_id} | Product: {pid:10} | CF: {cf_s:.4f} | CB: {cb_s:.4f} | FreshBoost: {boost} | Final: {final_score:.4f}")

            blended.append({
                "productId": pid,
                "score": round(final_score, 6),
                "cb_score": round(cb_s, 6),
                "cf_score": round(cf_s, 6),
            })

        blended.sort(key=lambda x: x["score"], reverse=True)
        return blended[:top_k]

    def trending_fallback(self, top_k: int = config.RANKING_TOP_K) -> list[dict]:
        """Return top-K trending items (used for fully cold-start users)."""
        return [
            {"productId": pid, "score": 0.0, "cb_score": 0.0, "cf_score": 0.0}
            for pid in self._trending[:top_k]
        ]
