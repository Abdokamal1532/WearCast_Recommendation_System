# =============================================================================
# src/evaluator.py
# -----------------------------------------------------------------------------
# Offline evaluation of recommendation quality.
#
# Protocol: Leave-One-Out (LOO)
#   • For each user, hold out their highest-weighted interaction as the "truth".
#   • Train on the remaining interactions.
#   • Predict top-K and check whether the held-out item appears.
#
# Metrics computed:
#   • Precision@K   — what fraction of top-K are relevant
#   • Recall@K      — what fraction of relevant items appear in top-K
#   • NDCG@K        — quality of ranking (penalizes relevant items ranked low)
# =============================================================================

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.models.content_based import ContentBasedModel
from src.models.collaborative import CollaborativeModel
from src.models.hybrid import HybridModel


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for pid in top_k if pid in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for pid in top_k if pid in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain @ K."""
    top_k = recommended[:k]
    dcg = 0.0
    for rank, pid in enumerate(top_k, start=1):
        if pid in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    # Ideal DCG: all relevant items at the top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Train/Test split
# ---------------------------------------------------------------------------

def train_test_split(
    interactions_df: pd.DataFrame,
    test_fraction: float = config.TEST_FRACTION,
    random_seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split: for each user, randomly hold out *test_fraction* of
    their interactions (minimum 1 item) as the test set.
    """
    rng = np.random.default_rng(random_seed)
    train_rows, test_rows = [], []

    for uid, grp in interactions_df.groupby("userId"):
        grp = grp.sample(frac=1, random_state=int(rng.integers(0, 1_000_000)))
        n_test = max(1, int(len(grp) * test_fraction))
        test_rows.append(grp.iloc[:n_test])
        train_rows.append(grp.iloc[n_test:])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs offline evaluation for CB, CF, and Hybrid models.

    Parameters
    ----------
    interactions_df : pd.DataFrame  — full interaction matrix
    item_features_df : pd.DataFrame
    user_profiles_df : pd.DataFrame
    k : int                         — evaluation cut-off
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
        user_profiles_df: pd.DataFrame,
        k: int = config.EVAL_K,
    ):
        self.interactions_df = interactions_df
        self.item_features_df = item_features_df
        self.user_profiles_df = user_profiles_df
        self.k = k

    def _evaluate_recommender(
        self,
        recommend_fn,
        test_df: pd.DataFrame,
        label: str,
    ) -> dict[str, float]:
        """
        Generic evaluation loop.

        Parameters
        ----------
        recommend_fn : callable(user_id, top_k) → list[str]
        test_df : pd.DataFrame  — held-out interactions
        """
        # Build ground truth: userId → set of held-out productIds
        ground_truth: dict[str, set[str]] = (
            test_df.groupby("userId")["productId"].apply(set).to_dict()
        )

        p_scores, r_scores, n_scores = [], [], []
        users = list(ground_truth.keys())

        for uid in tqdm(users, desc=f"  Evaluating [{label}]", leave=False):
            relevant = ground_truth[uid]
            try:
                recommended = recommend_fn(uid, self.k)
            except Exception:
                recommended = []

            p_scores.append(precision_at_k(recommended, relevant, self.k))
            r_scores.append(recall_at_k(recommended, relevant, self.k))
            n_scores.append(ndcg_at_k(recommended, relevant, self.k))

        return {
            f"Precision@{self.k}": round(float(np.mean(p_scores)), 4),
            f"Recall@{self.k}": round(float(np.mean(r_scores)), 4),
            f"NDCG@{self.k}": round(float(np.mean(n_scores)), 4),
        }

    def run(self) -> dict[str, dict[str, float]]:
        """
        Full evaluation pipeline:
          1. Split interactions into train / test
          2. Re-train CB and CF on train split
          3. Evaluate CB, CF, Hybrid against test split
        """
        print("\n[Evaluator] Splitting interactions into train/test...")
        train_df, test_df = train_test_split(self.interactions_df)
        print(f"[Evaluator] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

        # --- Train Content-Based on train split ---
        print("[Evaluator] Fitting Content-Based model on training data...")
        # Rebuild user profiles on train data only
        from src.preprocessor import Preprocessor
        p = Preprocessor.__new__(Preprocessor)
        p.interactions_df = train_df
        p.item_features_df = self.item_features_df
        p.reference_date = __import__("datetime").datetime.strptime(
            config.REFERENCE_DATE, "%Y-%m-%d"
        )
        train_user_profiles = p.build_user_profiles()
        cb = ContentBasedModel(self.item_features_df, train_user_profiles)

        def cb_recommend(user_id, k):
            return [pid for pid, _ in cb.recommend(user_id, top_k=k)]

        # --- Train Collaborative on train split ---
        print("[Evaluator] Training LightFM model on training data...")
        cf = CollaborativeModel(train_df)
        cf.train()

        def cf_recommend(user_id, k):
            return [pid for pid, _ in cf.recommend(user_id, top_k=k)]

        # --- Hybrid ---
        hybrid = HybridModel(cb, cf, train_df)

        def hybrid_recommend(user_id, k):
            return [r["productId"] for r in hybrid.recommend(user_id, top_k=k)]

        # --- Evaluate ---
        print("\n[Evaluator] Running evaluation...")
        results = {}
        results["Content-Based"] = self._evaluate_recommender(cb_recommend, test_df, "CB")
        results["Collaborative"] = self._evaluate_recommender(cf_recommend, test_df, "CF")
        results["Hybrid"] = self._evaluate_recommender(hybrid_recommend, test_df, "Hybrid")

        return results


def print_results(results: dict[str, dict[str, float]]) -> None:
    """Pretty-print evaluation results to stdout."""
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    header = f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'NDCG':>10}"
    print(header)
    print("-" * 55)
    for model_name, metrics in results.items():
        values = list(metrics.values())
        row = f"{model_name:<20} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}"
        print(row)
    print("=" * 55 + "\n")
