import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# API
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"  # swap to "llama-3.3-70b-versatile" for higher quality

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
SUPPORT_ISSUES_DIR = ROOT_DIR / "support_tickets"
INPUT_CSV = SUPPORT_ISSUES_DIR / "support_tickets.csv"
SAMPLE_CSV = SUPPORT_ISSUES_DIR / "sample_support_tickets.csv"
OUTPUT_CSV = SUPPORT_ISSUES_DIR / "output.csv"

# Corpus subdirectories
CORPUS_DIRS = {
    "hackerrank": DATA_DIR / "hackerrank",
    "claude": DATA_DIR / "claude",
    "visa": DATA_DIR / "visa",
}

# Retrieval
TOP_K_RESULTS = 5
MIN_SIMILARITY_SCORE = 0.1   # TF-IDF scores are lower than dense embeddings; 0.1 is reasonable
CHUNK_SIZE = 500              # characters per chunk
CHUNK_OVERLAP = 100

# ChromaDB
CHROMA_COLLECTION_NAME = "support_corpus"

# Allowed output values
ALLOWED_STATUS = {"replied", "escalated"}
ALLOWED_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
