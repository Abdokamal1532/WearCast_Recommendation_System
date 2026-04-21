import json
import pandas as pd
import os

def verify():
    print("[Verification] Loading data...")
    with open('data/raw/events.json', 'r') as f:
        data = json.load(f)
    
    events = data['events']
    catalog = data['catalog']
    
    # 1. Find a user with Filter events
    filter_users = [e['userId'] for e in events if e['eventType'] == 'filter']
    if not filter_users:
        print("[FAIL] No filter events found in raw data.")
        return
    
    test_user = filter_users[0]
    user_filters = [e for e in events if e['userId'] == test_user and e['eventType'] == 'filter']
    
    print(f"[INFO] Test User: {test_user}")
    print(f"[INFO] Found {len(user_filters)} filter(s) for this user.")
    for f in user_filters[:2]:
        print(f"  - Filter: {f['filters']}")

    # 2. Check their profile
    profiles_df = pd.read_csv('data/processed/user_profiles.csv')
    user_profile = profiles_df[profiles_df['userId'] == test_user]
    
    if user_profile.empty:
        print(f"[FAIL] No profile found for user {test_user}")
        return
        
    print("[INFO] User Profile (Top Features):")
    # Get columns with non-zero values
    vec = user_profile.iloc[0].drop('userId')
    top_features = vec[vec > 0].sort_values(ascending=False).head(5)
    print(top_features)
    
    # 3. Validation: Does the profile match the filters?
    # (Checking if the top feature matches a filtered category)
    first_filter_cat = user_filters[0]['filters'].get('categoryName')
    if first_filter_cat:
        feat_col = f"cat_{first_filter_cat}"
        if feat_col in vec and vec[feat_col] > 0:
            print(f"[SUCCESS] User profile correctly contains intent feature: {feat_col} (Value: {vec[feat_col]:.4f})")
        else:
            print(f"[WARNING] Feature {feat_col} not prominent in profile.")

if __name__ == "__main__":
    verify()
