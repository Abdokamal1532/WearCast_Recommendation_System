# =============================================================================
# src/db_loader.py
# -----------------------------------------------------------------------------
# Fetches real catalog and activity data from WearCast SQL Server DB.
# Maps it to the format expected by the recommendation system.
# =============================================================================

import json
import os
import sys
from datetime import datetime, timedelta
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Enum Mapping (based on .NET source)
# TargetAudience: Men=1, Women=2, Unisex=3, Kids=4, Babies=8
TARGET_AUDIENCE_MAP = {
    1: "MEN",
    2: "WOMEN",
    3: "UNISEX",
    4: "KIDS",
    8: "BABIES"
}

# DressStyle: Casual=1, Formal=2, Party=3, Gym=4, Sporty=5
DRESS_STYLE_MAP = {
    1: "CASUAL",
    2: "FORMAL",
    3: "PARTY",
    4: "GYM",
    5: "SPORTY"
}

def get_case_insensitive(d, key, default=None):
    """Truly case-insensitive dictionary lookup."""
    if not isinstance(d, dict): return default
    # Create a lower-case mapping
    lower_map = {str(k).lower(): v for k, v in d.items()}
    return lower_map.get(str(key).lower(), default)

def fetch_from_db():
    print(f"[DataLoader] Connecting to database...")
    
    # Check if we have the necessary driver/setup
    if "pyodbc" in config.DB_CONNECTION_STRING and "mssql" in config.DB_CONNECTION_STRING:
        try:
            import pyodbc
        except ImportError:
            print("[DataLoader] [ERR] 'pyodbc' is not installed. SQL Server connection skipped.")
            raise RuntimeError("pyodbc not installed")

    try:
        engine = create_engine(config.DB_CONNECTION_STRING)
    except Exception as e:
        print(f"[DataLoader] [ERR] Could not create engine: {e}")
        raise e
    
    with engine.connect() as conn:
        # 1. Fetch Categories
        print("[DataLoader] [DB] Querying Category table...")
        categories_df = pd.read_sql("SELECT Id, Name FROM Categories", conn)
        category_map = dict(zip(categories_df['Id'], categories_df['Name']))
        print(f"[DataLoader] [META] Mapped {len(category_map)} categories.")
        
        # 2. Fetch Catalog (Designed + Fixed)
        print("[DataLoader] [DB] Extracting Designed and Fixed products catalog...")
        
        # Designed Products
        designed_query = """
            SELECT p.Id, p.Name, p.Price, p.TargetAudience, p.DressStyle, p.CategoryId, p.FactoryId as SellerId, p.CreatedOn
            FROM DesignedProducts p WHERE p.IsDeleted = 0
        """
        d_df = pd.read_sql(designed_query, conn)
        d_df['Type'] = "DESIGNED"
        d_df['IdPrefix'] = "DES_"
        print(f"[DataLoader] [META] Found {len(d_df)} Designed Products.")
        
        # Fixed Products
        fixed_query = """
            SELECT p.Id, p.Name, p.Price, p.TargetAudience, p.DressStyle, p.CategoryId, p.SellerId, p.CreatedOn
            FROM FixedProducts p WHERE p.IsDeleted = 0
        """
        f_df = pd.read_sql(fixed_query, conn)
        f_df['Type'] = "FIXED"
        f_df['IdPrefix'] = "FIX_"
        print(f"[DataLoader] [META] Found {len(f_df)} Fixed Products.")
        
        # Combine
        products_df = pd.concat([d_df, f_df], ignore_index=True)
        
        catalog = []
        for _, row in products_df.iterrows():
            pid = f"{row['IdPrefix']}{row['Id']}"
            catalog.append({
                "productId": pid,
                "productType": row['Type'],
                "price": float(row['Price']),
                "targetAudience": TARGET_AUDIENCE_MAP.get(row['TargetAudience'], "UNISEX"),
                "dressStyle": DRESS_STYLE_MAP.get(row['DressStyle'], "CASUAL"),
                "categoryName": category_map.get(row['CategoryId'], "Unknown"),
                "sellerId": str(row['SellerId']),
                "createdOn": row['CreatedOn'].isoformat() if hasattr(row['CreatedOn'], 'isoformat') else str(row['CreatedOn'])
            })
            
        # 3. Fetch Activity Logs
        lookback_date = datetime.now() - timedelta(days=config.DB_LOOKBACK_DAYS)
        logs_query = f"""
            SELECT UserId, Payload, CreatedAt 
            FROM UserActivityLogs 
            WHERE CreatedAt >= '{lookback_date.strftime('%Y-%m-%d %H:%M:%S')}'
        """
        print(f"[DataLoader] [DB] Fetching activity logs (since {lookback_date.strftime('%Y-%m-%d')})...")
        logs_df = pd.read_sql(logs_query, conn)
        print(f"[DataLoader] [RAW] Found {len(logs_df)} raw log entries in database.")
        
        events = []
        parsed_counts = {"filter": 0, "click": 0, "purchase": 0, "view": 0, "addtocart": 0}

        for _, row in logs_df.iterrows():
            try:
                payload = json.loads(row['Payload'])
                event_type = str(get_case_insensitive(payload, 'eventType', 'unknown')).lower()
                
                # Check for legacy structure if eventType is missing
                if event_type == "unknown":
                    if get_case_insensitive(payload, 'Filters'): event_type = 'filter'
                    elif get_case_insensitive(payload, 'ProductDetails'): event_type = 'click'
                    elif get_case_insensitive(payload, 'Products'): event_type = 'purchase'
                
                event = {
                    "userId": str(row['UserId']),
                    "timestamp": row['CreatedAt'].isoformat() + "Z",
                    "eventType": event_type
                }
                
                if event_type == 'filter':
                    f_data = get_case_insensitive(payload, 'Filters', {})
                    event["filters"] = {
                        "categoryName": get_case_insensitive(f_data, "CategoryName"),
                        "dressStyle": get_case_insensitive(f_data, "DressStyle"),
                        "targetAudience": get_case_insensitive(f_data, "TargetAudience")
                    }
                elif event_type in ['click', 'view', 'addtocart']:
                    details = get_case_insensitive(payload, 'ProductDetails', {})
                    pid = get_case_insensitive(details, "productId")
                    if pid:
                        event["productDetails"] = {"productId": str(pid)}
                elif event_type == 'purchase':
                    p_list = get_case_insensitive(payload, 'Products', [])
                    items = []
                    for p in (p_list if isinstance(p_list, list) else []):
                        pid = get_case_insensitive(p, "productId")
                        if pid: items.append({"productId": str(pid)})
                    if items: event["products"] = items
                
                if event["eventType"] in parsed_counts:
                    parsed_counts[event["eventType"]] += 1
                    events.append(event)
            except:
                continue
                
        print(f"[DataLoader] [PARSE_STATS] Done. Breakdown: {parsed_counts}")
    return catalog, events

def get_db_fingerprint():
    """Returns a unique fingerprint of the DB state (row counts)."""
    try:
        engine = create_engine(config.DB_CONNECTION_STRING)
        with engine.connect() as conn:
            logs_count = conn.execute(text("SELECT COUNT(*) FROM UserActivityLogs")).scalar()
            designed_count = conn.execute(text("SELECT COUNT(*) FROM DesignedProducts")).scalar()
            fixed_count = conn.execute(text("SELECT COUNT(*) FROM FixedProducts")).scalar()
            return f"L{logs_count}_D{designed_count}_F{fixed_count}"
    except Exception as e:
        print(f"[DataLoader] [ERR] Failed to check DB fingerprint: {e}")
        return None

def sync():
    catalog, events = fetch_from_db()
    
    os.makedirs(config.RAW_DIR, exist_ok=True)
    output_path = config.RAW_EVENTS_PATH
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"events": events, "catalog": catalog}, f, indent=2, ensure_ascii=False)
    
    print(f"[DataLoader] Written data to {output_path}")

if __name__ == "__main__":
    sync()
