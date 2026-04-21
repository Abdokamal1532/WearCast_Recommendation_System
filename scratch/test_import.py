import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from src.api import app
    print("SUCCESS: API imported correctly")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
