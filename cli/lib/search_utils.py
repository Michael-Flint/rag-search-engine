import json
import os
from typing import Any

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_LIMIT = 200
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 1

DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
BM25_K1 = 1.5   #BM25 TermFreq saturation tuning factor
BM25_B = 0.75   #BM25 DocumentLength normalisation factor (Longer documents are penalised, shorter documents are boosted)




def format_search_result(doc_id: str, title: str, document: str, score: float, **metadata: Any) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }



def load_movies() -> list[dict]:
    data_path = os.path.join(PROJECT_ROOT, "data", "movies.json")
    with open(data_path, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[dict]:
    stopwords_path = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
    with open(stopwords_path) as f:        
        return f.read().splitlines()