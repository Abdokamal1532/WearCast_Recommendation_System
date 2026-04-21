# =============================================================================
# src/preprocessor.py
# -----------------------------------------------------------------------------
# Reads raw events.json, cleans the data, applies time-decay weighting,
# builds User-Item interaction matrix, item feature matrix, and user profiles.
#
# Outputs (CSV):
#   data/processed/interactions.csv   — (userId, productId, weight)
#   data/processed/item_features.csv  — one-hot encoded item attributes
#   data/processed/user_profiles.csv  — weighted-average of item features per user
# =============================================================================

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_since(timestamp_str: str, reference: datetime) -> float:
    """Return number of days between *timestamp_str* and *reference*."""
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        return max((reference - dt).total_seconds() / 86_400, 0.0)
    except ValueError:
        return 0.0


def _time_decay(days: float) -> float:
    """Exponential time-decay factor: exp(-λ × days)."""
    return float(np.exp(-config.DECAY_LAMBDA * days))


def _normalize_audience(value) -> list[str]:
    """Ensure targetAudience is always a list of upper-cased strings."""
    if isinstance(value, list):
        return [str(v).upper() for v in value]
    if value is None:
        return ["UNKNOWN"]
    return [str(value).upper()]


# ---------------------------------------------------------------------------
# Event parsers — each returns list of (userId, productId, raw_weight, timestamp)
# ---------------------------------------------------------------------------

def _parse_purchase(event: dict) -> list[tuple]:
    rows = []
    uid = event.get("userId", "unknown")
    ts = event.get("timestamp", "")
    for product in event.get("products", []):
        pid = product.get("productId")
        if pid:
            rows.append((uid, pid, config.EVENT_WEIGHTS["purchase"], ts))
    return rows


def _parse_click(event: dict) -> list[tuple]:
    uid = event.get("userId", "unknown")
    ts = event.get("timestamp", "")
    details = event.get("productDetails", {})
    pid = details.get("productId")
    if pid:
        return [(uid, pid, config.EVENT_WEIGHTS["click"], ts)]
    return []


def _parse_view(event: dict) -> list[tuple]:
    uid = event.get("userId", "unknown")
    ts = event.get("timestamp", "")
    details = event.get("productDetails", {})
    pid = details.get("productId")
    if pid:
        return [(uid, pid, config.EVENT_WEIGHTS["view"], ts)]
    return []


def _parse_addtocart(event: dict) -> list[tuple]:
    uid = event.get("userId", "unknown")
    ts = event.get("timestamp", "")
    details = event.get("productDetails", {})
    pid = details.get("productId")
    if pid:
        return [(uid, pid, config.EVENT_WEIGHTS["addToCart"], ts)]
    return []


def _parse_filter(event: dict) -> list[dict]:
    """
    Filters express strong explicit intent.
    We return a list of intent dictionaries to be used for user profiling.
    """
    uid = event.get("userId", "unknown")
    ts = event.get("timestamp", "")
    filters = event.get("filters", {})
    
    # We create a "virtual feature vector" based on the filters
    intent = {
        "userId": uid,
        "timestamp": ts,
        "weight": config.EVENT_WEIGHTS.get("filter", 5.0), # Stronger weight
        "features": {
            "categoryName": filters.get("categoryName"),
            "dressStyle": filters.get("dressStyle"),
            "targetAudience": _normalize_audience(filters.get("targetAudience"))
        }
    }
    
    # Only return if at least one filter is present
    if any(intent["features"].values()):
        return [intent]
    return []


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Orchestrates all preprocessing steps:
      1. Load raw events + catalog
      2. Extract interactions with time-decay weights
      3. Build item feature matrix (one-hot)
      4. Build user profiles (weighted avg of item features)
    """

    def __init__(self, raw_path: str = config.RAW_EVENTS_PATH):
        self.raw_path = raw_path
        self.reference_date = datetime.strptime(config.REFERENCE_DATE, "%Y-%m-%d")

        self.interactions_df: pd.DataFrame | None = None
        self.item_features_df: pd.DataFrame | None = None
        self.user_profiles_df: pd.DataFrame | None = None
        self.catalog: list[dict] = []
        self.all_product_ids: list[str] = []
        self.all_user_ids: list[str] = []

    # ------------------------------------------------------------------
    # Step 1 – Load & parse
    # ------------------------------------------------------------------

    def load_raw(self) -> tuple[list[dict], list[dict]]:
        print(f"[Preprocessor] [SYSTEM] Starting raw data load from: {self.raw_path}")
        with open(self.raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("events", [])
        catalog = data.get("catalog", [])
        print(f"[Preprocessor] [SUCCESS] Loaded metadata: {len(events):,} events and {len(catalog)} catalog items.")
        
        # Split events into interactions and intent
        self.raw_interactions = []
        self.raw_intents = []
        for e in events:
            if e.get("eventType") == "filter":
                self.raw_intents.append(e)
            else:
                self.raw_interactions.append(e)
        
        print(f"[Preprocessor] [INFO] Split: {len(self.raw_interactions)} direct interactions, {len(self.raw_intents)} filter intents.")
        return events, catalog

    # ------------------------------------------------------------------
    # Step 2 – Build interaction matrix
    # ------------------------------------------------------------------

    def build_interactions(self, events: list[dict]) -> pd.DataFrame:
        """
        Parse all events → (userId, productId, weight) rows.
        Multiple interactions for the same (user, item) pair are summed.
        """
        print("[Preprocessor] Building interaction matrix...")
        rows = []
        parsers = {
            "purchase": _parse_purchase,
            "click": _parse_click,
            "view": _parse_view,
            "addtocart": _parse_addtocart,
            "filter": _parse_filter,
        }

        for event in self.raw_interactions:
            etype = event.get("eventType", "unknown").lower()
            parser = parsers.get(etype)
            if parser is None:
                continue
            
            raw_rows = parser(event)
            ts = event.get("timestamp", "")
            days = _days_since(ts, self.reference_date)
            decay = _time_decay(days)
            for uid, pid, weight, _ in raw_rows:
                rows.append({
                    "userId": uid,
                    "productId": pid,
                    "weight": weight * decay,
                })

        df = pd.DataFrame(rows)
        if df.empty:
            print("[Preprocessor] [WARNING] Interaction list is empty. Creating empty DataFrame.")
            self.interactions_df = pd.DataFrame(columns=["userId", "productId", "weight"])
            self.all_user_ids = []
            return self.interactions_df

        # Aggregate duplicate (user, item) pairs
        df = (
            df.groupby(["userId", "productId"], as_index=False)["weight"]
            .sum()
        )
        df["weight"] = df["weight"].clip(lower=0.01)  # ensure positive

        self.all_user_ids = sorted(df["userId"].unique().tolist())
        self.interactions_df = df
        print(f"[Preprocessor] Interaction matrix: {len(df):,} rows, "
              f"{df['userId'].nunique()} users, {df['productId'].nunique()} items.")
        return df

    # ------------------------------------------------------------------
    # Step 3 – Item feature matrix
    # ------------------------------------------------------------------

    def build_item_features(self, catalog: list[dict]) -> pd.DataFrame:
        """
        One-hot encode: categoryName, dressStyle, targetAudience (multi-label).
        Normalize price to [0, 1].
        """
        print("[Preprocessor] Building item feature matrix...")

        if not catalog:
            print("[Preprocessor] [WARNING] Catalog is empty. Creating dummy feature matrix.")
            self.item_features_df = pd.DataFrame(columns=["productId", "price_norm"])
            self.all_product_ids = []
            return self.item_features_df

        records = []
        for item in catalog:
            records.append({
                "productId": item["productId"],
                "categoryName": item.get("categoryName", "Unknown"),
                "dressStyle": item.get("dressStyle", "Unknown"),
                "targetAudience": _normalize_audience(item.get("targetAudience", "UNKNOWN")),
                "price": float(item.get("price", 0)),
                "sellerId": item.get("sellerId") or "unknown",
            })

        df = pd.DataFrame(records)
        self.all_product_ids = df["productId"].tolist()

        # Normalize price
        p_max = df["price"].max()
        df["price_norm"] = df["price"] / p_max if p_max > 0 else 0.0

        # One-hot: categoryName
        cat_dummies = pd.get_dummies(df["categoryName"], prefix="cat")

        # One-hot: dressStyle
        style_dummies = pd.get_dummies(df["dressStyle"], prefix="style")

        # Multi-label binarize: targetAudience
        mlb = MultiLabelBinarizer()
        audience_matrix = mlb.fit_transform(df["targetAudience"])
        audience_df = pd.DataFrame(
            audience_matrix,
            columns=[f"aud_{c}" for c in mlb.classes_],
        )

        self.item_features_df = pd.concat(
            [df[["productId", "price_norm"]], cat_dummies, style_dummies, audience_df],
            axis=1,
        ).reset_index(drop=True)

        print(f"[Preprocessor] Item features: {len(self.item_features_df)} items, "
              f"{self.item_features_df.shape[1] - 1} feature columns.")
        return self.item_features_df

    # ------------------------------------------------------------------
    # Step 4 – User profile matrix
    # ------------------------------------------------------------------

    def build_user_profiles(self) -> pd.DataFrame:
        """
        User profile = weighted average of feature vectors of items the user
        interacted with (weighted by interaction weight).
        """
        feature_cols = [c for c in self.item_features_df.columns if c != "productId"]
        print(f"[Preprocessor] [PROFILING] Starting profile generation for {self.item_features_df.shape[0]} items across {len(feature_cols)} features.")
        # Factor 1: Item-based profile (from clicks/purchases)
        item_feat = self.item_features_df.set_index("productId")[feature_cols]

        profiles = {}
        seen_weight = {}
        # 1. Process Interactions
        print(f"[Preprocessor] [STAGE 1] Processing {len(self.interactions_df)} interaction rows...")
        for uid, grp in self.interactions_df.groupby("userId"):
            weights = grp["weight"].values
            total_weight = weights.sum()
            matched = item_feat.reindex(grp["productId"].values).fillna(0.0)
            weighted_sum = matched.values * weights[:, np.newaxis]
            profiles[uid] = weighted_sum.sum(axis=0)
            seen_weight[uid] = total_weight
        print(f"[Preprocessor] [OK] Direct interaction profiles built for {len(profiles)} users.")
            
        # 2. Process Intent (Filters)
        print(f"[Preprocessor] [STAGE 2] Injecting {len(self.raw_intents)} high-fidelity filter intents...")
        intent_count = 0
        for event in self.raw_intents:
            intent_rows = _parse_filter(event)
            for intent in intent_rows:
                uid = intent["userId"]
                days = _days_since(intent["timestamp"], self.reference_date)
                weight = intent["weight"] * _time_decay(days)
                
                intent_count += 1
                # Convert features to vector
                feat = intent["features"]
                vec = np.zeros(len(feature_cols))
                
                # Category boost
                if feat["categoryName"]:
                    col = f"cat_{feat['categoryName']}"
                    if col in feature_cols: vec[feature_cols.index(col)] = 1.0
                
                # Style boost
                if feat["dressStyle"]:
                    col = f"style_{feat['dressStyle']}"
                    if col in feature_cols: vec[feature_cols.index(col)] = 1.0
                    
                # Audience boost (multi-label)
                for aud in feat["targetAudience"]:
                    col = f"aud_{aud}"
                    if col in feature_cols: vec[feature_cols.index(col)] = 1.0
                
                if uid not in profiles:
                    profiles[uid] = np.zeros(len(feature_cols))
                    # Initialize seen_weight if needed (using a local dict to track per-user weight)
                
                profiles[uid] += vec * weight
                # Track total weight for normalization
                # Since we want to normalize later, we'll store totals in a separate dict
        
        # 3. Normalize all profiles
        rows = []
        for uid, vec in profiles.items():
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            row = {"userId": uid}
            for i, col in enumerate(feature_cols):
                row[col] = vec[i]
            rows.append(row)

        profiles_df = pd.DataFrame(rows)

        self.user_profiles_df = profiles_df
        print(f"[Preprocessor] User profiles: {len(profiles_df)} users.")
        return profiles_df

    # ------------------------------------------------------------------
    # Step 5 – Save to disk
    # ------------------------------------------------------------------

    def save(self):
        """Persist all processed DataFrames to data/processed/."""
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        self.interactions_df.to_csv(config.INTERACTIONS_PATH, index=False)
        self.item_features_df.to_csv(config.ITEM_FEATURES_PATH, index=False)
        self.user_profiles_df.to_csv(config.USER_PROFILES_PATH, index=False)
        print(f"[Preprocessor] Saved processed files -> {config.PROCESSED_DIR}")

    # ------------------------------------------------------------------
    # Convenience: run all steps
    # ------------------------------------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Complete preprocessing pipeline: Clean -> Build Matrix -> Build Profiles."""
        print("\n" + "="*60)
        print("[Preprocessor] Starting Full AI Ingestion Stack")
        print("="*60)
        
        events, catalog = self.load_raw()
        self.catalog = catalog
        
        print(f"[Preprocessor] [STEP 1/3] Building Interaction Matrix...")
        self.build_interactions(events)
        
        print(f"[Preprocessor] [STEP 2/3] Generating Item Feature Map...")
        self.build_item_features(catalog)
        
        print(f"[Preprocessor] [STEP 3/3] Constructing Intent-Aware User Profiles...")
        self.build_user_profiles()
        
        self.save()
        print("[Preprocessor] [FINISHED] Pipeline execution complete.")
        print("="*60 + "\n")
        return self.interactions_df, self.item_features_df, self.user_profiles_df


# ---------------------------------------------------------------------------
# Loader helpers (used by models and API)
# ---------------------------------------------------------------------------

def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load already-processed CSVs from disk."""
    interactions = pd.read_csv(config.INTERACTIONS_PATH)
    item_features = pd.read_csv(config.ITEM_FEATURES_PATH)
    user_profiles = pd.read_csv(config.USER_PROFILES_PATH)
    return interactions, item_features, user_profiles


if __name__ == "__main__":
    p = Preprocessor()
    p.run()
