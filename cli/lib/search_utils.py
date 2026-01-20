import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))



def load_movies() -> list[dict]:
    data_path = os.path.join(PROJECT_ROOT, "data", "movies.json")
    with open(data_path, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[dict]:
    stopwords_path = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
    with open(stopwords_path) as f:        
        return f.read().splitlines()