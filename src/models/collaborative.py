# Collaborative Filtering via TruncatedSVD (Matrix Factorisation).
#
# Because compilation of C++ extensions like LightFM can fail on Windows
# without the proper build tools, we use scikit-learn's robust TruncatedSVD
# on the sparse User-Item interaction matrix. This provides an excellent
# and fully cross-platform baseline for collaborative filtering.

import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class CollaborativeModel:
    """
    SVD-based Collaborative Filtering recommender.

    Parameters
    ----------
    interactions_df : pd.DataFrame
        Columns: [userId, productId, weight]
    """

    def __init__(self, interactions_df: pd.DataFrame):
        self.interactions_df = interactions_df.copy()
        
        self.model: TruncatedSVD | None = None
        self.interaction_matrix: sp.csr_matrix | None = None
        
        # User/Item embedding matrices
        self.user_embeddings: np.ndarray | None = None
        self.item_embeddings: np.ndarray | None = None

        self._user_id_map: dict[str, int] = {}
        self._item_id_map: dict[str, int] = {}
        self._inv_item_map: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Build sparse interaction matrix
    # ------------------------------------------------------------------

    def _build_dataset(self) -> None:
        """Construct sparse CSR matrix from the interactions DataFrame."""
        unique_users = sorted(self.interactions_df["userId"].unique())
        unique_items = sorted(self.interactions_df["productId"].unique())

        self._user_id_map = {uid: i for i, uid in enumerate(unique_users)}
        self._item_id_map = {pid: i for i, pid in enumerate(unique_items)}
        self._inv_item_map = {i: pid for pid, i in self._item_id_map.items()}

        row_indices = self.interactions_df["userId"].map(self._user_id_map).values
        col_indices = self.interactions_df["productId"].map(self._item_id_map).values
        weights = self.interactions_df["weight"].values

        self.interaction_matrix = sp.csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(len(unique_users), len(unique_items)),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> "CollaborativeModel":
        """
        Fit TruncatedSVD to learn latent representations of users and items.
        Returns self for method chaining.
        """
        print("[CollaborativeModel] Building interaction matrix...")
        self._build_dataset()

        # Guard against zero interactions
        if self.interaction_matrix.nnz == 0:
            print("[CollaborativeModel] [WARNING] No interactions to train on. Skipping SVD.")
            return self

        # The number of components shouldn't exceed the number of users/items
        n_components = min(
            config.LIGHTFM_COMPONENTS, 
            self.interaction_matrix.shape[0] - 1,
            self.interaction_matrix.shape[1] - 1
        )

        if n_components < 1:
            print("[CollaborativeModel] [WARNING] Not enough users/items for SVD. Skipping.")
            return self

        print(
            f"[CollaborativeModel] Training TruncatedSVD proxy for CF "
            f"(components={n_components})..."
        )

        self.model = TruncatedSVD(
            n_components=n_components,
            random_state=config.RANDOM_SEED,
            n_iter=7,
        )

        # Factorise: R ≈ U * Sigma * V^T
        # We can treat user embeddings as U * sqrt(Sigma) and item embeddings as V * sqrt(Sigma)
        user_factors = self.model.fit_transform(self.interaction_matrix)  # U * Sigma
        
        # self.model.components_ is V^T. We transpose to get V.
        # We'll multiply by sqrt(Sigma) to balance the latent space slightly.
        # For simple dot products, user_factors @ model.components_ accurately reconstructs the matrix.
        self.user_embeddings = user_factors
        self.item_embeddings = self.model.components_.T

        print("[CollaborativeModel] Training complete [OK]")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_product_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return top-K (productId, score) for *user_id*.
        Returns empty list for unknown (cold-start) users.
        """
        if self.model is None or self.user_embeddings is None:
            raise RuntimeError("Model not trained. Call train() first.")

        user_idx = self._user_id_map.get(user_id)
        if user_idx is None:
            return []   # cold-start

        # Reconstruct user's scores for all items: user_vec @ item_matrix^T
        user_vec = self.user_embeddings[user_idx]
        scores = user_vec @ self.item_embeddings.T

        exclude = set(exclude_product_ids or [])
        results = [
            (self._inv_item_map[i], float(scores[i]))
            for i in range(len(self._inv_item_map))
            if self._inv_item_map[i] not in exclude
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def score_items(self, user_id: str, product_ids: list[str]) -> dict[str, float]:
        """
        Score a specific list of product_ids for a user.
        Used by the hybrid model.
        """
        if self.model is None or self.user_embeddings is None:
            raise RuntimeError("Model not trained. Call train() first.")

        user_idx = self._user_id_map.get(user_id)
        if user_idx is None:
            return {pid: 0.0 for pid in product_ids}

        user_vec = self.user_embeddings[user_idx]
        
        scores = {}
        for pid in product_ids:
            item_idx = self._item_id_map.get(pid)
            if item_idx is not None:
                item_vec = self.item_embeddings[item_idx]
                scores[pid] = float(np.dot(user_vec, item_vec))
            else:
                scores[pid] = 0.0
        return scores

    def get_all_product_ids(self) -> list[str]:
        return list(self._item_id_map.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = config.LIGHTFM_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "user_embeddings": self.user_embeddings,
                    "item_embeddings": self.item_embeddings,
                    "user_id_map": self._user_id_map,
                    "item_id_map": self._item_id_map,
                    "inv_item_map": self._inv_item_map,
                },
                f,
            )
        print(f"[CollaborativeModel] SVD Model saved -> {path}")

    @classmethod
    def load(cls, interactions_df: pd.DataFrame, path: str = config.LIGHTFM_MODEL_PATH) -> "CollaborativeModel":
        instance = cls(interactions_df)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance.model = data["model"]
        instance.user_embeddings = data["user_embeddings"]
        instance.item_embeddings = data["item_embeddings"]
        instance._user_id_map = data["user_id_map"]
        instance._item_id_map = data["item_id_map"]
        instance._inv_item_map = data["inv_item_map"]
        print(f"[CollaborativeModel] SVD Model loaded <- {path}")
        return instance
