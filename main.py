from src.api import app

# This file serves as an entrypoint for Vercel.
# It redirects to the FastAPI application defined in src/api.py.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
