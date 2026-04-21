# =============================================================================
# src/retrieval.py
# -----------------------------------------------------------------------------
# Stage 1 of the two-stage pipeline: fast candidate retrieval.
#
# Goal: Narrow the full catalog (300 items) down to a small candidate set
# (~100 items) using the Content-Based model as a lightweight filter.
# This is fast to compute and significantly reduces the ranking workload.
#
# Heuristic: union of
#   • Top-K CB items (personalized)
#   • Top trending items (as diversity buffer)
# =============================================================================

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.models.content_based import ContentBasedModel


class CandidateRetriever:
    """
    Fast candidate retrieval using Content-Based cosine similarity.

    Parameters
    ----------
    cb_model : ContentBasedModel
    interactions_df : pd.DataFrame
        Used to build the global trending list.
    retrieval_top_k : int
        Number of CB candidates to retrieve (default from config).
    """

    def __init__(
        self,
        cb_model: ContentBasedModel,
        interactions_df: pd.DataFrame,
        retrieval_top_k: int = config.RETRIEVAL_TOP_K,
    ):
        self.cb_model = cb_model
        self.retrieval_top_k = retrieval_top_k

        # Global trending: top items by total interaction weight
        self._trending: list[str] = (
            interactions_df.groupby("productId")["weight"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

        # User interaction history for exclusion (optional — can re-recommend)
        self._user_history: dict[str, set[str]] = (
            interactions_df.groupby("userId")["productId"]
            .apply(set)
            .to_dict()
        )

    def retrieve(
        self,
        user_id: str,
        exclude_seen: bool = False,
    ) -> list[str]:
        """
        Retrieve up to *retrieval_top_k* candidate product IDs for *user_id*.

        Parameters
        ----------
        user_id : str
        exclude_seen : bool
            If True, already-interacted items are excluded.

        Returns
        -------
        list[str] — deduplicated candidate product IDs.
        """
        exclude = self._user_history.get(user_id, set()) if exclude_seen else set()

        # CB candidates
        cb_results = self.cb_model.recommend(
            user_id=user_id,
            top_k=self.retrieval_top_k,
            exclude_product_ids=list(exclude),
        )
        cb_candidates = [pid for pid, _ in cb_results]

        # Trending buffer (add diversity)
        diversity_k = max(self.retrieval_top_k // 5, 10)
        trending_buffer = [p for p in self._trending if p not in exclude][:diversity_k]

        # Merge and deduplicate (preserve CB-first ordering)
        seen_pids: set[str] = set()
        candidates: list[str] = []
        for pid in cb_candidates + trending_buffer:
            if pid not in seen_pids:
                candidates.append(pid)
                seen_pids.add(pid)

        return candidates[: self.retrieval_top_k]
