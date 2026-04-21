# =============================================================================
# run_pipeline.py
# -----------------------------------------------------------------------------
# Single-command end-to-end pipeline runner.
#
# Usage:
#   python run_pipeline.py              # full pipeline
#   python run_pipeline.py --skip-gen   # skip data generation (reuse existing data)
#   python run_pipeline.py --skip-eval  # skip evaluation (faster — just train & save)
# =============================================================================

import argparse
import os
import sys
import time
from datetime import datetime
import json

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

BANNER = r"""
==========================================================
                                                          
        RECOMMENDATION SYSTEM - PROOF OF CONCEPT         
        Hybrid Content-Based + Collaborative Filtering    
                                                          
==========================================================
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Recommendation System PoC Pipeline")
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help="Skip data generation (use existing data/raw/events.json)",
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        default=True,
        help="Fetch real catalog and events from SQL Server (DEFAULT)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip offline evaluation",
    )
    parser.add_argument(
        "--schedule",
        type=int,
        help="Run in continuous schedule mode (interval in hours)",
        default=None
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Smart Mode: watches DB for changes and re-trains automatically",
    )
    return parser.parse_args()


def step(label: str):
    """Print a formatted step header."""
    print(f"\n{'-' * 60}")
    print(f"  STEP: {label}")
    print(f"{'-' * 60}")


def _notify_reload():
    """Notify the FastAPI service to reload models."""
    try:
        import requests
        reload_url = os.getenv("RELOAD_URL", "http://localhost:8000/internal/reload")
        print(f"  [SIGNAL] Notifying API to reload models at {reload_url}...")
        resp = requests.post(reload_url, timeout=10)
        if resp.status_code == 200:
            print("  [SIGNAL] [OK] API reloaded successfully.")
        else:
            print(f"  [SIGNAL] [FAIL] API returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"  [SIGNAL] [ERR] Could not notify API: {e}")

def main():
    args = parse_args()
    
    if args.watch:
        print("="*60)
        print("  SMART WATCH MODE ENABLED")
        print("  Monitoring SQL Server for new users/products...")
        print("="*60)
        from src.db_loader import get_db_fingerprint
        
        last_fp = None
        while True:
            current_fp = get_db_fingerprint()
            
            if current_fp and current_fp != last_fp:
                if last_fp is not None:
                    print(f"\n[{datetime.now()}] [CHANGE DETECTED] DB State shifted: {last_fp} -> {current_fp}")
                    run_pipeline(args)
                    _notify_reload()
                else:
                    # First run
                    run_pipeline(args)
                    _notify_reload()
                
                last_fp = current_fp
            
            # Poll every 30 seconds
            time.sleep(30)
            
    elif args.schedule:
        print("="*60)
        print(f"  CONTINUOUS TRAINING MODE ENABLED")
        print(f"  Interval: Every {args.schedule} hours")
        print("="*60)
        
        while True:
            # Re-run main logic but with schedule=None to avoid recursion
            # Just call a dedicated runner function or run once
            run_pipeline(args)
            _notify_reload()
            
            print(f"\n[{datetime.now()}] Next training in {args.schedule} hours...")
            time.sleep(args.schedule * 3600)
    else:
        run_pipeline(args)

def run_pipeline(args):
    print(BANNER)
    total_start = time.time()

    # STEP 1 — Real Data Synchronization (SQL Server)
    # -----------------------------------------------------------------------
    if args.from_db:
        step("1/5  Data Ingestion")
        from src.db_loader import sync
        print("[Main] [DB] Synchronizing with WearCast SQL Server...")
        sync()
        
        # Reload the synced data to show stats in Main
        with open(config.RAW_EVENTS_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            events = raw_data.get("events", [])
            catalog = raw_data.get("catalog", [])
        
        print(f"  [OK] Successfully ingested {len(catalog)} products.")
        print(f"  [OK] Successfully ingested {len(events)} activity logs.")
    elif not args.skip_gen:
        step("1/5  Data Generation")
        from src.data_generator import generate
        t0 = time.time()
        events, catalog = generate()
        print(f"  [OK] Generated {len(events):,} events, {len(catalog)} catalog items  "
              f"({time.time() - t0:.1f}s)")
    else:
        print("\n  [INFO] Skipping data generation — using existing events.json")
        import json
        with open(config.RAW_EVENTS_PATH) as f:
            data = json.load(f)
        events = data["events"]
        catalog = data["catalog"]

    # -----------------------------------------------------------------------
    # STEP 2 — Preprocess
    # -----------------------------------------------------------------------
    step("2/5  Preprocessing")
    t0 = time.time()
    from src.preprocessor import Preprocessor
    preprocessor = Preprocessor()
    interactions_df, item_features_df, user_profiles_df = preprocessor.run()
    print(f"  [OK] Preprocessing complete ({time.time() - t0:.1f}s)")
    print(f"  |  Users: {interactions_df['userId'].nunique()}")
    print(f"  |  Items: {interactions_df['productId'].nunique()}")
    print(f"  |  Interactions: {len(interactions_df):,}")
    print(f"  |  Item feature columns: {item_features_df.shape[1] - 1}")

    # -----------------------------------------------------------------------
    # STEP 3 — Train Content-Based Model
    # -----------------------------------------------------------------------
    step("3/5  Training Content-Based Model")
    t0 = time.time()
    from src.models.content_based import ContentBasedModel
    cb_model = ContentBasedModel(item_features_df, user_profiles_df)
    print(f"  [OK] Content-Based model ready ({time.time() - t0:.1f}s)")

    # -----------------------------------------------------------------------
    # STEP 4 — Train Collaborative Filtering Model (TruncatedSVD)
    # -----------------------------------------------------------------------
    step("4/5  Training Collaborative Filtering Model")
    t0 = time.time()
    from src.models.collaborative import CollaborativeModel
    cf_model = CollaborativeModel(interactions_df)
    
    if len(interactions_df) > 0:
        try:
            cf_model.train()
            cf_model.save()
            print(f"  [OK] CF model trained & saved ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  [WARN] CF Training failed: {e}. Model will not be saved.")
    else:
        print("  [WARN] Skipping CF training: No interactions found in database.")

    # Build Hybrid
    from src.models.hybrid import HybridModel
    hybrid_model = HybridModel(cb_model, cf_model, interactions_df, catalog)

    # Build pipeline components
    from src.retrieval import CandidateRetriever
    from src.ranking import Ranker
    retriever = CandidateRetriever(cb_model, interactions_df)
    ranker = Ranker(hybrid_model, item_features_df, preprocessor.catalog)

    # -----------------------------------------------------------------------
    # STEP 5 — Evaluation
    # -----------------------------------------------------------------------
    if not args.skip_eval and len(interactions_df) > 10:
        step("5/5  Offline Evaluation")
        try:
            t0 = time.time()
            from src.evaluator import Evaluator, print_results
            evaluator = Evaluator(interactions_df, item_features_df, user_profiles_df)
            results = evaluator.run()
            print_results(results)
            print(f"  [OK] Evaluation complete ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  [WARN] Evaluation failed: {e}")
    else:
        print("\n  [INFO] Skipping evaluation (insufficient data or --skip-eval)")

    # -----------------------------------------------------------------------
    # Demo: sample recommendations for a random user
    # -----------------------------------------------------------------------
    if len(interactions_df) > 0:
        step("DEMO  Sample Recommendations")
        import random
        random.seed(config.RANDOM_SEED)
        sample_users = interactions_df["userId"].unique().tolist()
        demo_user = random.choice(sample_users)

        print(f"\n  Recommendations for: {demo_user}\n")
        print(f"  {'Rank':<5} {'ProductID':<12} {'Score':>8}  {'CB':>8}  {'CF':>8}  "
              f"{'Category':<14} {'Style':<12} {'Price':>7}")
        print(f"  {'-' * 80}")

        candidates = retriever.retrieve(demo_user, exclude_seen=True)
        ranked = ranker.rank(demo_user, candidates, top_k=10)

        for i, item in enumerate(ranked, 1):
            audiences = item.get("targetAudience", "")
            if isinstance(audiences, list):
                audiences = "/".join(audiences)
            print(
                f"  {i:<5} {item['productId']:<12} "
                f"{item['score']:>8.4f}  "
                f"{item['cb_score']:>8.4f}  "
                f"{item['cf_score']:>8.4f}  "
                f"{str(item.get('categoryName', '')):.<14} "
                f"{str(item.get('dressStyle', '')):.<12} "
                f"${item.get('price', 0):>6}"
            )
    else:
        print("\n  [INFO] No interactions found. Skipping demo recommendations.")

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  [SUCCESS]  Pipeline complete in {total_elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"\n  To serve the API:")
    print(f"    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000\n")


if __name__ == "__main__":
    main()
