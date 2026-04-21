# =============================================================================
# config.py — Central configuration for the Recommendation System PoC
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

RAW_EVENTS_PATH = os.path.join(RAW_DIR, "events.json")
INTERACTIONS_PATH = os.path.join(PROCESSED_DIR, "interactions.csv")
USER_PROFILES_PATH = os.path.join(PROCESSED_DIR, "user_profiles.csv")
ITEM_FEATURES_PATH = os.path.join(PROCESSED_DIR, "item_features.csv")
LIGHTFM_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lightfm_model.pkl")
ITEM_FEATURES_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "item_features_matrix.npz")

# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------
NUM_USERS = 500
NUM_ITEMS = 300
NUM_EVENTS = 50_000
RANDOM_SEED = 42

# Event distribution (must sum to 1.0)
EVENT_DISTRIBUTION = {
    "filter": 0.60,
    "click": 0.30,
    "purchase": 0.10,
}

# Product catalog constants
CATEGORIES = ["Dresses", "Hoodies", "Shirts", "Pants", "Shoes", "Jackets", "Skirts", "Accessories"]
DRESS_STYLES = ["FORMAL", "CASUAL", "STREETWEAR", "SPORTS", "ELEGANT"]
TARGET_AUDIENCES = ["WOMEN", "MEN", "UNISEX", "KIDS"]
PRICE_RANGES = {
    "Dresses": (150, 800),
    "Hoodies": (100, 600),
    "Shirts": (50, 400),
    "Pants": (80, 500),
    "Shoes": (120, 900),
    "Jackets": (200, 1200),
    "Skirts": (80, 450),
    "Accessories": (20, 300),
}
NUM_SELLERS = 50

# ---------------------------------------------------------------------------
# Event Weights  (implicit feedback signal strength)
# ---------------------------------------------------------------------------
EVENT_WEIGHTS = {
    "purchase": 5.0,
    "addToCart": 3.0,
    "click": 2.0,
    "view": 1.0,
    "filter": 10.0,
}

# ---------------------------------------------------------------------------
# Time Decay
# ---------------------------------------------------------------------------
# weight *= exp(-DECAY_LAMBDA * days_since_event)
# λ=0.005 → event at 30 days ago ≈ 86% weight, 180 days ago ≈ 41% weight
DECAY_LAMBDA = 0.005

# Reference timestamp for time-decay calculations (use current run time)
REFERENCE_DATE = "2026-04-14"   # matches current seeding date

# ---------------------------------------------------------------------------
# Model Hyper-parameters
# ---------------------------------------------------------------------------
# LightFM
LIGHTFM_LOSS = "warp"           # Weighted Approximate Rank Pairwise — best for implicit
LIGHTFM_COMPONENTS = 20         # Latent dimension (reduced to prevent overfitting on small datasets)
LIGHTFM_EPOCHS = 30
LIGHTFM_NUM_THREADS = 4
LIGHTFM_LEARNING_RATE = 0.05
LIGHTFM_ITEM_ALPHA = 1e-6       # L2 regularisation on item embeddings
LIGHTFM_USER_ALPHA = 1e-6       # L2 regularisation on user embeddings

# Hybrid blending  (alpha=1 → pure CF,  alpha=0 → pure CB)
HYBRID_ALPHA = 0.6

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = 100   # candidates fetched in Stage 1
RANKING_TOP_K = 10      # final recommendations returned to user

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
TEST_FRACTION = 0.20    # 20 % of interactions held out for evaluation
EVAL_K = 10             # Precision@K, Recall@K, NDCG@K

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEFAULT_TOP_K = 10

# ---------------------------------------------------------------------------
# Database (Integration with WearCast API)
# ---------------------------------------------------------------------------
# Update this connection string if your SQL Server instance/driver is different
# In production, set the DB_PASSWORD environment variable
DB_LOOKBACK_DAYS = 1  # Only ingest interactions from the last 1 day (for speed/relevance)

DB_CONNECTION_STRING = os.getenv(
    "DB_CONNECTION_STRING",
    "mssql+pyodbc://localhost\\MSSQLSERVER01/WearCastDb?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes&TrustServerCertificate=yes"
)
