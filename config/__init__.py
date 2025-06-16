import os
from dotenv import load_dotenv

# Load environment variables from .env
ENV_PATH = os.path.abspath(os.path.join("./config/.env"))
load_dotenv(dotenv_path=ENV_PATH)

# Helper to get env vars safely
def get_env(key: str, default=None) -> str:
    return os.getenv(key, default)