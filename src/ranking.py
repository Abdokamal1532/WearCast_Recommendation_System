# =============================================================================
# src/ranking.py
# -----------------------------------------------------------------------------
# Stage 2 of the two-stage pipeline: re-rank candidates using the Hybrid model
# and apply optional filter constraints.
#
# Filter constraints (from user's last filter event):
#   • minPrice / maxPrice
#   • targetAudience
#   • dressStyle
#   • categoryName
#   • sellerId
# =============================================================================

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.models.hybrid import HybridModel


class Ranker:
    """
    Re-ranks a candidate list using the Hybrid model, then applies hard
    filter constraints to ensure results match the user's active filters.

    Parameters
    ----------
    hybrid_model : HybridModel
    item_features_df : pd.DataFrame
        Full item catalog with attributes for filter matching.
    """

    def __init__(self, hybrid_model: HybridModel, item_features_df: pd.DataFrame, catalog: list[dict]):
        self.hybrid_model = hybrid_model
        # Build a fast lookup: productId → catalog dict
        self._catalog_map: dict[str, dict] = {item["productId"]: item for item in catalog}
        self._item_features_df = item_features_df.set_index("productId")

    # ------------------------------------------------------------------
    # Filter application
    # ------------------------------------------------------------------

    def _passes_filters(self, product: dict, filters: dict) -> bool:
        """Return True if *product* satisfies all non-null *filters*."""
        if not filters:
            return True

        # Price range
        price = product.get("price", 0)
        if filters.get("minPrice") is not None and price < filters["minPrice"]:
            return False
        if filters.get("maxPrice") is not None and price > filters["maxPrice"]:
            return False

        # Target audience (product must overlap with requested audiences)
        if filters.get("targetAudience"):
            req_audiences = (
                filters["targetAudience"]
                if isinstance(filters["targetAudience"], list)
                else [filters["targetAudience"]]
            )
            req_audiences = [a.upper() for a in req_audiences]
            prod_audiences = product.get("targetAudience", [])
            if isinstance(prod_audiences, str):
                prod_audiences = [prod_audiences]
            prod_audiences = [a.upper() for a in prod_audiences]
            if not any(a in prod_audiences for a in req_audiences):
                return False

        # Dress style
        if filters.get("dressStyle"):
            if product.get("dressStyle", "").upper() != filters["dressStyle"].upper():
                return False

        # Category
        if filters.get("categoryName"):
            if product.get("categoryName", "").lower() != filters["categoryName"].lower():
                return False

        # Seller
        if filters.get("sellerId"):
            if product.get("sellerId") != filters["sellerId"]:
                return False

        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        user_id: str,
        candidates: list[str],
        top_k: int = config.RANKING_TOP_K,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Re-rank *candidates* for *user_id* and return top-K results.

        Parameters
        ----------
        user_id : str
        candidates : list[str]
            Product IDs from the retrieval stage.
        top_k : int
        filters : dict | None
            Optional filter constraints. Keys: minPrice, maxPrice,
            targetAudience, dressStyle, categoryName, sellerId.

        Returns
        -------
        list of dicts:
            [{productId, score, cb_score, cf_score, price, categoryName,
              dressStyle, targetAudience, sellerId}, ...]
        """
        # Get hybrid scores for all candidates
        scored = self.hybrid_model.recommend(
            user_id=user_id,
            top_k=len(candidates),  # score all candidates, filter after
            candidate_pool=candidates,
        )

        # Enrich with catalog metadata + apply filters
        results = []
        for item in scored:
            pid = item["productId"]
            product = self._catalog_map.get(pid)
            if product is None:
                continue

            if filters and not self._passes_filters(product, filters):
                continue

            results.append({
                "productId": pid,
                "score": item["score"],
                "cb_score": item["cb_score"],
                "cf_score": item["cf_score"],
                "price": product.get("price"),
                "categoryName": product.get("categoryName"),
                "dressStyle": product.get("dressStyle"),
                "targetAudience": product.get("targetAudience"),
                "sellerId": product.get("sellerId"),
                "productType": product.get("productType"),
            })

        # If filters removed too many results, pad from trending
        if len(results) < top_k and filters:
            trending_fallback = self.hybrid_model.trending_fallback(top_k * 3)
            seen_pids = {r["productId"] for r in results}
            for item in trending_fallback:
                pid = item["productId"]
                if pid in seen_pids:
                    continue
                product = self._catalog_map.get(pid)
                if product is None:
                    continue
                if not self._passes_filters(product, filters):
                    continue
                results.append({
                    "productId": pid,
                    "score": item["score"],
                    "cb_score": 0.0,
                    "cf_score": 0.0,
                    "price": product.get("price"),
                    "categoryName": product.get("categoryName"),
                    "dressStyle": product.get("dressStyle"),
                    "targetAudience": product.get("targetAudience"),
                    "sellerId": product.get("sellerId"),
                    "productType": product.get("productType"),
                })
                if len(results) >= top_k:
                    break

        return results[:top_k]
