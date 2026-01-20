#import json
import os
import pickle
#import re
from collections import defaultdict
from .keyword_search import tokenize_text
from .search_utils import load_movies

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class InvertedIndex:
    #def __init__(self, index, docmap):
    def __init__(self):        
        self.index = defaultdict(set)   # Dictionary that maps tokens (strings) to sets of document IDs (integers)
        self.docmap = {}                # Dictionary that maps document IDs (int) to their full document object
        #self.index = index      
        #self.docmap = docmap    



    def __add_document(self, doc_id, text):
        # Tokenise the input text
        tokens = tokenize_text(text)
        
        # Store the full document
        self.docmap[doc_id] = text
        
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
            self._InvertedIndex__add_document(doc_id, text)



    def save(self):
        # Save the index and docmap attributes to disk using the pickle modules dump function
        
        cache_path = os.path.join(PROJECT_ROOT, "cache")
        # Create the cache directory if it doesn't exist
        os.makedirs(cache_path, exist_ok=True)

        # File paths
        index_path = os.path.join(cache_path, "index.pkl")
        docmap_path = os.path.join(cache_path, "docmap.pkl")

        # Save index
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        # Save docmap
        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)        

