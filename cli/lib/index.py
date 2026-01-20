import os
import pickle
from collections import defaultdict
from .keyword_search import tokenize_text
from .search_utils import load_movies, CACHE_PATH



class InvertedIndex:
    def __init__(self):        
        self.index = defaultdict(set)   # Dictionary that maps tokens (strings) to sets of document IDs (integers)
        self.docmap = {}                # Dictionary that maps document IDs (int) to their full document object



    def __add_document(self, doc_id, text):
        # Tokenise the input text
        tokens = tokenize_text(text)
        
        # Add each token to the index with the document ID
        for token in tokens:
            self.index[token].add(doc_id)
        


    def get_documents(self, token: str):
        # Get the set of document ID's for a given token (set it to lowercase)
        # Return them as a list, sorted in ascending order
        token = token.lower()
        return sorted(self.index.get(token, set()))
    


    def build(self):
        # Iterate over all of the movies and add them to both the index and the docmap            
        # Load movie data
        movies = load_movies()

        # Iterate over all movies and add them to the index and docmap
        for m in movies:
            doc_id = m["id"]
            # When adding the movie data to the index with __add_document(), concatenate the title and the description and use that as the input text
            text = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, text)


    def load(self):
        # Load the index and docmap from disk using pickle
        # File paths
        index_path = os.path.join(CACHE_PATH, "index.pkl")
        docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        
        # Raise an error if the files don't exist
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"Docmap file not found: {docmap_path}")

        # Load index
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        # Load docmap
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        
    


    def save(self):
        # Save the index and docmap attributes to disk using the pickle modules dump function
        # Create the cache directory if it doesn't exist
        os.makedirs(CACHE_PATH, exist_ok=True)

        # File paths
        index_path = os.path.join(CACHE_PATH, "index.pkl")
        docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")

        # Save index
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        # Save docmap
        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)        

