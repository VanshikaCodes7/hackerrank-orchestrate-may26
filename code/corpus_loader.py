"""
corpus_loader.py
Loads support documents from data/{hackerrank,claude,visa}/,
chunks them, and builds a TF-IDF index.
Caches the index to disk so subsequent runs load instantly.
"""

import re
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    CORPUS_DIRS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Cache file location — sits next to corpus_loader.py
CACHE_PATH = Path(__file__).parent / ".tfidf_cache.pkl"


def _corpus_fingerprint() -> str:
    """Generate a fingerprint of all corpus files to detect changes."""
    hasher = hashlib.md5()
    supported_extensions = {".md", ".txt", ".html", ".htm", ".csv"}
    for company, directory in sorted(CORPUS_DIRS.items()):
        if not directory.exists():
            continue
        for filepath in sorted(directory.rglob("*")):
            if filepath.is_file() and filepath.suffix.lower() in supported_extensions:
                stat = filepath.stat()
                hasher.update(f"{filepath}:{stat.st_size}:{stat.st_mtime}".encode())
    return hasher.hexdigest()


def _read_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return ""


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sentence

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 50]


def load_corpus() -> List[Dict]:
    supported_extensions = {".md", ".txt", ".html", ".htm", ".csv"}
    all_chunks = []

    for company, directory in CORPUS_DIRS.items():
        if not directory.exists():
            print(f"  [WARN] Corpus directory not found: {directory}")
            continue

        files = [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        print(f"  Loading {company}: {len(files)} files found")
        company_count = 0

        for filepath in files:
            text = _read_file(filepath)
            if not text:
                continue

            chunks = _chunk_text(text)
            for idx, chunk in enumerate(chunks):
                raw = f"{company}::{filepath}::{idx}"
                chunk_id = hashlib.md5(raw.encode()).hexdigest()
                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk,
                    "company": company,
                    "source_file": str(filepath.name),
                })
                company_count += 1

        print(f"    -> {company_count} chunks indexed for {company}")

    print(f"\n  Total chunks: {len(all_chunks)}")
    return all_chunks


def build_tfidf_index(chunks: List[Dict]) -> Tuple:
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, chunks


def initialize_corpus() -> Tuple:
    """
    Load corpus and build TF-IDF index.
    Uses a disk cache — rebuilds only if corpus files have changed.
    """
    fingerprint = _corpus_fingerprint()

    # Try loading from cache
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                print("[Corpus Loader] Cache hit — loaded index from disk.\n")
                return None, cached["vectorizer"], cached["matrix"], cached["chunks"]
            else:
                print("[Corpus Loader] Corpus changed — rebuilding index...")
        except Exception as e:
            print(f"[Corpus Loader] Cache read failed ({e}) — rebuilding...")

    # Build fresh
    print("\n[Corpus Loader] Loading corpus files...")
    chunks = load_corpus()

    if not chunks:
        raise RuntimeError(
            "No corpus chunks loaded. Check that data/ directories exist and contain files."
        )

    print("[Corpus Loader] Building TF-IDF index...")
    vectorizer, matrix, chunks = build_tfidf_index(chunks)

    # Save to cache
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({
                "fingerprint": fingerprint,
                "vectorizer": vectorizer,
                "matrix": matrix,
                "chunks": chunks,
            }, f)
        print(f"[Corpus Loader] Index cached to {CACHE_PATH.name}")
    except Exception as e:
        print(f"[Corpus Loader] Cache write failed (non-fatal): {e}")

    print("[Corpus Loader] All indexes ready.\n")
    return None, vectorizer, matrix, chunks
