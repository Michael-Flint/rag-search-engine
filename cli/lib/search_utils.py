import json
import os

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_LIMIT = 200

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
BM25_K1 = 1.5   #BM25 TermFreq saturation tuning factor
BM25_B = 0.75   #BM25 DocumentLength normalisation factor (Longer documents are penalised, shorter documents are boosted)




def load_movies() -> list[dict]:
    data_path = os.path.join(PROJECT_ROOT, "data", "movies.json")
    with open(data_path, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[dict]:
    stopwords_path = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
    with open(stopwords_path) as f:        
        return f.read().splitlines()