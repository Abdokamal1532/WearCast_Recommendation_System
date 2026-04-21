# =============================================================================
# src/data_generator.py
# -----------------------------------------------------------------------------
# Generates a realistic synthetic event log (purchase / click / filter) that
# mirrors the JSON schema provided in the project brief.
#
# Output : data/raw/events.json
# =============================================================================

import json
import os
import random
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dirs():
    os.makedirs(config.RAW_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)


def _random_date(start_days_ago: int = 180, end_days_ago: int = 0) -> str:
    """Return a random ISO-8601 timestamp between *start_days_ago* and *end_days_ago*."""
    ref = datetime(2026, 4, 10, 23, 59, 59)
    delta_days = random.randint(end_days_ago, start_days_ago)
    delta_secs = random.randint(0, 86_400)
    dt = ref - timedelta(days=delta_days, seconds=delta_secs)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_product_catalog(rng: np.random.Generator) -> list[dict]:
    """Create a fixed catalog of NUM_ITEMS products."""
    catalog = []
    for i in range(config.NUM_ITEMS):
        category = rng.choice(config.CATEGORIES)
        price_min, price_max = config.PRICE_RANGES[category]
        price = int(rng.integers(price_min, price_max + 1))

        # Some items target multiple audiences (realistic)
        n_audiences = rng.choice([1, 1, 1, 2], p=[0.6, 0.2, 0.1, 0.1])
        audiences = rng.choice(config.TARGET_AUDIENCES, size=n_audiences, replace=False).tolist()

        product = {
            "productId": f"prod_{i:04d}",
            "productType": rng.choice(["DESIGNED", "STANDARD", "PREMIUM"], p=[0.3, 0.5, 0.2]),
            "price": price,
            "quantity": int(rng.integers(1, 100)),
            "targetAudience": audiences if len(audiences) > 1 else audiences[0],
            "dressStyle": rng.choice(config.DRESS_STYLES),
            "categoryName": category,
            "sellerId": rng.choice(
                [f"seller_{j:03d}" for j in range(config.NUM_SELLERS)] + [None],
                p=[0.95 / config.NUM_SELLERS] * config.NUM_SELLERS + [0.05],
            ) if rng.random() < 0.05 else f"seller_{int(rng.integers(0, config.NUM_SELLERS)):03d}",
        }
        catalog.append(product)
    return catalog


def _make_user_seeds(rng: np.random.Generator) -> dict[str, dict]:
    """
    Each user gets a 'preference seed' that biases which products they
    interact with (makes the event log non-random, so the model has signal).
    """
    seeds = {}
    for i in range(config.NUM_USERS):
        user_id = f"user_{i:04d}"
        seeds[user_id] = {
            "preferred_categories": rng.choice(
                config.CATEGORIES,
                size=rng.integers(1, 4),
                replace=False,
            ).tolist(),
            "preferred_styles": rng.choice(
                config.DRESS_STYLES,
                size=rng.integers(1, 3),
                replace=False,
            ).tolist(),
            "preferred_audiences": rng.choice(
                config.TARGET_AUDIENCES,
                size=rng.integers(1, 3),
                replace=False,
            ).tolist(),
            "max_price": int(rng.choice([300, 500, 800, 1200, 2000])),
        }
    return seeds


def _score_product_for_user(product: dict, seed: dict) -> float:
    """
    Compute a soft affinity score [0, 1] between a user seed and a product.
    Used to weight the product sampling so users interact with relevant items.
    """
    score = 0.0
    if product["categoryName"] in seed["preferred_categories"]:
        score += 0.4
    if product["dressStyle"] in seed["preferred_styles"]:
        score += 0.3
    audiences = (
        product["targetAudience"]
        if isinstance(product["targetAudience"], list)
        else [product["targetAudience"]]
    )
    if any(a in seed["preferred_audiences"] for a in audiences):
        score += 0.2
    if product["price"] <= seed["max_price"]:
        score += 0.1
    return score


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------

def _build_purchase_event(user_id: str, products: list[dict], rng: np.random.Generator) -> dict:
    """Create a purchase event (1–3 products)."""
    n = int(rng.integers(1, 4))
    chosen = rng.choice(products, size=min(n, len(products)), replace=False).tolist()
    items = []
    for p in chosen:
        items.append({
            "productId": p["productId"],
            "productType": p["productType"],
            "price": p["price"],
            "quantity": int(rng.integers(1, 4)),
            "targetAudience": p["targetAudience"],
            "dressStyle": p["dressStyle"],
            "categoryName": p["categoryName"],
            "sellerId": p["sellerId"],
        })
    return {
        "eventType": "purchase",
        "userId": user_id,
        "timestamp": _random_date(),
        "products": items,
    }


def _build_click_event(user_id: str, product: dict, rng: np.random.Generator) -> dict:
    """Create a click/view event for a single product."""
    audiences = (
        product["targetAudience"]
        if isinstance(product["targetAudience"], list)
        else [product["targetAudience"]]
    )
    return {
        "eventType": "click",
        "userId": user_id,
        "timestamp": _random_date(),
        "productDetails": {
            "productId": product["productId"],
            "price": product["price"],
            "targetAudience": audiences,
            "dressStyle": product["dressStyle"],
            "categoryName": product["categoryName"],
            "sellerId": product["sellerId"],
        },
    }


def _build_filter_event(user_id: str, seed: dict, rng: np.random.Generator) -> dict:
    """Create a search/filter event grounded in the user's seed preferences."""
    min_price = int(rng.integers(0, 300))
    max_price = min(seed["max_price"], int(rng.integers(400, 2000)))
    n_audiences = int(rng.integers(1, 3))
    audiences = rng.choice(seed["preferred_audiences"], size=min(n_audiences, len(seed["preferred_audiences"])), replace=False).tolist()

    return {
        "eventType": "filter",
        "userId": user_id,
        "timestamp": _random_date(),
        "filters": {
            "searchKey": rng.choice(["", "", "", rng.choice(config.CATEGORIES)]),
            "minPrice": min_price,
            "maxPrice": max_price,
            "targetAudience": audiences,
            "dressStyle": rng.choice(seed["preferred_styles"]),
            "categoryName": rng.choice(seed["preferred_categories"]),
            "sellerId": rng.choice(
                [f"seller_{j:03d}" for j in range(config.NUM_SELLERS)] + [None]
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(output_path: str = config.RAW_EVENTS_PATH) -> list[dict]:
    """
    Generate NUM_EVENTS synthetic events and write them to *output_path*.

    Returns the list of event dicts.
    """
    _make_dirs()
    rng = np.random.default_rng(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print(f"[DataGenerator] Generating product catalog ({config.NUM_ITEMS} items)...")
    catalog = _make_product_catalog(rng)

    print(f"[DataGenerator] Building user preference seeds ({config.NUM_USERS} users)...")
    user_seeds = _make_user_seeds(rng)
    user_ids = list(user_seeds.keys())

    # Pre-compute per-user affinity weights over the catalog
    print("[DataGenerator] Computing user-item affinities...")
    affinity_cache: dict[str, np.ndarray] = {}
    for uid, seed in user_seeds.items():
        weights = np.array([_score_product_for_user(p, seed) for p in catalog], dtype=float)
        weights += 0.01          # ensure every item has a non-zero probability
        weights /= weights.sum()
        affinity_cache[uid] = weights

    # Determine event counts per type
    n_filter = int(config.NUM_EVENTS * config.EVENT_DISTRIBUTION["filter"])
    n_click = int(config.NUM_EVENTS * config.EVENT_DISTRIBUTION["click"])
    n_purchase = config.NUM_EVENTS - n_filter - n_click

    events: list[dict] = []

    # --- Filter events ---
    print(f"[DataGenerator] Generating {n_filter:,} filter events...")
    for _ in range(n_filter):
        uid = rng.choice(user_ids)
        events.append(_build_filter_event(uid, user_seeds[uid], rng))

    # --- Click events ---
    print(f"[DataGenerator] Generating {n_click:,} click events...")
    for _ in range(n_click):
        uid = rng.choice(user_ids)
        idx = rng.choice(len(catalog), p=affinity_cache[uid])
        events.append(_build_click_event(uid, catalog[idx], rng))

    # --- Purchase events ---
    print(f"[DataGenerator] Generating {n_purchase:,} purchase events...")
    for _ in range(n_purchase):
        uid = rng.choice(user_ids)
        # For purchases, sample 1–3 high-affinity products
        weights = affinity_cache[uid] ** 2   # sharpen distribution for purchases
        weights /= weights.sum()
        n_items = int(rng.integers(1, 4))
        idxs = rng.choice(len(catalog), size=min(n_items, len(catalog)), replace=False, p=weights)
        chosen = [catalog[i] for i in idxs]
        events.append(_build_purchase_event(uid, chosen, rng))

    # Shuffle chronologically (sort by timestamp for realism)
    rng.shuffle(events)

    print(f"[DataGenerator] Writing {len(events):,} events → {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"events": events, "catalog": catalog}, f, indent=2, ensure_ascii=False)

    print("  [OK] Data generation complete!")
    return events, catalog


if __name__ == "__main__":
    generate()
