"""
retriever.py
TF-IDF based retriever.
No PyTorch, no ONNX, no ChromaDB — pure sklearn, works on any OS.
"""

from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

from config import TOP_K_RESULTS, MIN_SIMILARITY_SCORE

# Keep context under ~3000 chars to stay within Groq free tier token limits
MAX_CONTEXT_CHARS = 3000


def retrieve(
    query: str,
    collection,
    vectorizer,
    matrix,
    chunks: List[Dict],
    company: Optional[str] = None,
    top_k: int = TOP_K_RESULTS,
) -> Dict:
    # Normalize company to lowercase for case-insensitive matching
    company_filter = company.lower().strip() if company and company.lower().strip() != "none" else None

    def _search(filter_company):
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, matrix).flatten()

        indices = [
            i for i, c in enumerate(chunks)
            if filter_company is None or c["company"].lower() == filter_company
        ]

        if not indices:
            indices = list(range(len(chunks)))

        scored = sorted([(i, scores[i]) for i in indices], key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        hits = []
        for idx, score in top:
            chunk = chunks[idx]
            hits.append({
                "text": chunk["text"],
                "company": chunk["company"],
                "source_file": chunk["source_file"],
                "similarity": round(float(score), 4),
            })
        return hits

    hits = _search(company_filter)
    best_score = max((h["similarity"] for h in hits), default=0.0)

    if best_score < MIN_SIMILARITY_SCORE and company_filter:
        hits = _search(None)
        best_score = max((h["similarity"] for h in hits), default=0.0)
        company_filter = None

    return {
        "hits": hits,
        "best_score": best_score,
        "has_good_match": best_score >= MIN_SIMILARITY_SCORE,
        "company_used": company_filter or "all",
    }


def format_context(hits: List[Dict]) -> str:
    """Format retrieved chunks, truncated to avoid Groq token limits."""
    if not hits:
        return "No relevant documentation found."

    parts = []
    total_chars = 0

    for i, hit in enumerate(hits, 1):
        entry = (
            f"[Doc {i} | Source: {hit['source_file']} | Score: {hit['similarity']}]\n"
            f"{hit['text']}"
        )
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                parts.append(entry[:remaining] + "...[truncated]")
            break
        parts.append(entry)
        total_chars += len(entry)

    return "\n\n---\n\n".join(parts)
