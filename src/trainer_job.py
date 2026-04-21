# =============================================================================
# src/trainer_job.py
# -----------------------------------------------------------------------------
# Production-grade background scheduler.
# This script runs in a loop, periodically triggering the AI training pipeline
# and notifying the FastAPI service to reload the updated models.
# =============================================================================

import os
import sys
import time
import requests
import subprocess
from datetime import datetime

# Set environment variables for the subprocess call
os.environ["PYTHONPATH"] = os.getcwd()

# Configuration
SYNC_INTERVAL_HOURS = int(os.getenv("SYNC_INTERVAL_HOURS", 1))
RELOAD_URL = os.getenv("RELOAD_URL", "https://wear-cast-recommendation-system-1.vercel.app/internal/reload")

def run_training_pipeline():
    """Executes the main training script as a subprocess."""
    print(f"\n[{datetime.now()}] [TRAINER] Starting automated sync and train...")
    
    try:
        # We run main.py as a subprocess to ensure clean memory and environment
        result = subprocess.run(
            [sys.executable, "main.py", "--from-db", "--skip-eval", "--skip-gen"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"[{datetime.now()}] [TRAINER] [SUCCESS] Pipeline complete.")
            # Print the summary stats from the output
            lines = result.stdout.splitlines()
            for line in lines:
                if "Breakdown" in line or "Pipeline complete" in line:
                    print(f"   {line}")
            return True
        else:
            print(f"[{datetime.now()}] [TRAINER] [ERROR] Pipeline failed.")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"[{datetime.now()}] [TRAINER] [CRITICAL] Subprocess crash: {e}")
        return False

def notify_api_reload():
    """Sends a signal to the FastAPI service to hot-reload the models."""
    print(f"[{datetime.now()}] [TRAINER] Notifying API to reload models...")
    try:
        response = requests.post(RELOAD_URL, timeout=10)
        if response.status_code == 200:
            print(f"[{datetime.now()}] [TRAINER] [RELOAD_OK] API acknowledged model refresh.")
        else:
            print(f"[{datetime.now()}] [TRAINER] [RELOAD_FAIL] API returned {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[{datetime.now()}] [TRAINER] [RELOAD_FAIL] Connection error: {e}")

def main_loop():
    print("="*60)
    print("  WEARCAST PRODUCTION AI SCHEDULER")
    print(f"  Internal Sync: Every {SYNC_INTERVAL_HOURS} hours")
    print("="*60)
    
    # Run once at startup
    if run_training_pipeline():
        notify_api_reload()
        
    while True:
        print(f"\n[{datetime.now()}] [SLEEP] Next sync in {SYNC_INTERVAL_HOURS} hours...")
        time.sleep(SYNC_INTERVAL_HOURS * 3600)
        
        if run_training_pipeline():
            notify_api_reload()

if __name__ == "__main__":
    main_loop()
