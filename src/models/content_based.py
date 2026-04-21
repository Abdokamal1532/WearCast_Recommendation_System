# =============================================================================
# src/models/content_based.py
# -----------------------------------------------------------------------------
# Content-Based Filtering model.
#
# Approach:
#   • Represent each item as a feature vector (one-hot + normalised price).
#   • Represent each user as the weighted average of their interacted items.
#   • Recommend items with highest cosine similarity to the user vector.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedModel:
    """
    Item-profile cosine-similarity recommender.

    Parameters
    ----------
    item_features_df : pd.DataFrame
        Columns: [productId, feature1, feature2, ...]
    user_profiles_df : pd.DataFrame
        Columns: [userId, feature1, feature2, ...]
    """

    def __init__(self, item_features_df: pd.DataFrame, user_profiles_df: pd.DataFrame):
        self.item_features_df = item_features_df.copy()
        self.user_profiles_df = user_profiles_df.copy()

        # Build numpy matrices for fast similarity computation
        self.product_ids: list[str] = item_features_df["productId"].tolist()
        self.feature_cols: list[str] = [
            c for c in item_features_df.columns if c != "productId"
        ]

        self.item_matrix: np.ndarray = item_features_df[self.feature_cols].values.astype(float)
        self._norm_item_matrix = self._l2_normalize(self.item_matrix)

        # Index user profiles for O(1) lookup
        self._user_index: dict[str, np.ndarray] = {}
        for _, row in user_profiles_df.iterrows():
            uid = row["userId"]
            vec = row[self.feature_cols].values.astype(float)
            self._user_index[uid] = vec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    def _get_user_vector(self, user_id: str) -> np.ndarray | None:
        return self._user_index.get(user_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_product_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return top-K (productId, score) pairs for *user_id*.

        Parameters
        ----------
        user_id : str
        top_k : int
        exclude_product_ids : list[str] | None
            Already-interacted items to exclude (avoid re-recommending).

        Returns
        -------
        list of (productId, score) sorted descending by score.
        """
        user_vec = self._get_user_vector(user_id)
        if user_vec is None:
            return []   # cold-start: caller handles fallback

        # Cosine similarity between user vector and all items
        user_norm = self._l2_normalize(user_vec.reshape(1, -1))
        scores = cosine_similarity(user_norm, self._norm_item_matrix)[0]  # shape (N_items,)

        # Build (pid, score) list and exclude already-seen
        exclude = set(exclude_product_ids or [])
        results = [
            (pid, float(score))
            for pid, score in zip(self.product_ids, scores)
            if pid not in exclude
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def score_items(self, user_id: str, product_ids: list[str]) -> dict[str, float]:
        """
        Return a dict {productId: cb_score} for a specific list of products.
        Used by the hybrid model to blend CB and CF scores.
        """
        user_vec = self._get_user_vector(user_id)
        if user_vec is None:
            return {pid: 0.0 for pid in product_ids}

        user_norm = self._l2_normalize(user_vec.reshape(1, -1))

        pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        scores = {}
        for pid in product_ids:
            idx = pid_to_idx.get(pid)
            if idx is not None:
                item_norm = self._norm_item_matrix[idx].reshape(1, -1)
                scores[pid] = float(cosine_similarity(user_norm, item_norm)[0][0])
            else:
                scores[pid] = 0.0
        return scores
