import requests
import json
import base64
import time

# Configuration
BASE_URL = "http://localhost:5245"
LOGIN_DATA = {
    "email": "Clifford.Nader47@hotmail.com",
    "password": "User123!"
}

def decode_and_print_token(token):
    try:
        parts = token.split('.')
        payload_b64 = parts[1]
        payload_b64 += '=' * (-len(payload_b64) % 4)
        payload = json.loads(base64.b64decode(payload_b64).decode('utf-8'))
        print("\n--- [🔐 JWT SECURITY ANALYSIS] ---")
        print(f"  User ID:   {payload.get('sub')}")
        print(f"  User Role: {payload.get('http://schemas.microsoft.com/ws/2008/06/identity/claims/role')}")
        print(f"  CustomerID:{payload.get('CustomerId')}")
        print("---------------------------------\n")
    except: pass

def test_system():
    print("====================================================")
    print("      WEARCAST AI SYSTEM - VERBOSE TEST")
    print("====================================================\n")

    # --- STEP 1: LOGIN ---
    print("--- [STEP 1] LOGGING INTO .NET API ---")
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/auth/login", json=LOGIN_DATA)
    latency = (time.time() - start_time) * 1000

    if response.status_code != 200:
        print(f"❌ Login Failed! Status: {response.status_code}")
        return

    full_response = response.json()
    inner_data = full_response.get("data", {})
    token = inner_data.get("token") or inner_data.get("Token")
    print(f"✅ [OK] Connected to .NET Identity (Latency: {latency:.0f}ms)")
    decode_and_print_token(token)

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    # --- STEP 2: TRACKING ---
    print("--- [STEP 2] SIMULATING USER INTEREST (LEARNING) ---")
    # Corrected URL path for product details based on controller route
    test_product_id = 200
    print(f"DEBUG: Viewing Designed Product ID {test_product_id}...")
    
    url = f"{BASE_URL}/api/catalog/designed-products/{test_product_id}"
    track_response = requests.get(url, headers=headers)
    
    if track_response.status_code == 200:
        print(f"✅ [OK] Product {test_product_id} viewed successfully.")
        print("       The .NET Backend has now logged this to SQL Server for the AI.")
    else:
        print(f"❌ Tracking failed with status {track_response.status_code}")
        print(f"URL tested: {url}")

    # --- STEP 3: RECOMMENDATIONS ---
    print("\n--- [STEP 3] FETCHING VERCEL AI RECOMMENDATIONS ---")
    print(f"DEBUG: Connecting to cloud inference server...")
    
    start_time = time.time()
    rec_response = requests.get(f"{BASE_URL}/api/customer-catalog/recommendations?topK=5", headers=headers)
    latency = (time.time() - start_time) * 1000

    print(f"--- [DEBUG] VERCEL CLOUD STATS ---")
    print(f"STATUS: {rec_response.status_code}")
    print(f"LATENCY: {latency:.0f}ms")
    print(f"SERVER: {rec_response.headers.get('Server')}")
    
    if rec_response.status_code == 200:
        print(f"\n✨ [SUCCESS] AI delivered personalized results!")
        print(json.dumps(rec_response.json(), indent=2))
    else:
        print(f"\n❌ FAILED to connect to Vercel AI: {rec_response.text}")

if __name__ == "__main__":
    test_system()
