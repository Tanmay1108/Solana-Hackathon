import sys
from pathlib import Path

# Ensure that the hack_solana directory is in the Python path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

import uvicorn
from app.main import app


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
