"""
Configuration settings for WriteWise AI - Email and Letter Assistant.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
HISTORY_DIR = DATA_DIR / "history"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB Settings
COLLECTION_NAME = "writing_assistant_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Default Chunking Settings
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# Default Retrieval Settings
DEFAULT_TOP_K = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.25

# LLM Providers and Models
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "allam-2-7b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
]
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Upload constraints
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = [".txt", ".pdf"]
