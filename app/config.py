"""
Centralised application configuration.

Loads environment variables once (via python-dotenv) and exposes them as a
single `settings` object so the rest of the codebase never calls
`os.getenv` directly. This is the standard "12-factor config" pattern and
is the kind of thing interviewers like to see: one source of truth for
config, easy to mock in tests, easy to reason about.
"""
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load a .env file if present (does nothing in production if env vars are
# already injected by the platform/container).
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    # --- LLM provider keys (CRAG uses Gemini -> Groq -> OpenAI as fallbacks) ---
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # --- Vector store ---
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db")
    )
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # --- App ---
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))


settings = Settings()
