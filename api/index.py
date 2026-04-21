import os
import sys

# Ensure project root is in PYTHONPATH
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from src.api import app

# This file serves as the entrypoint for Vercel.
