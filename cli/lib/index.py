import math
import os
import pickle
from collections import defaultdict, Counter
from .keyword_search import tokenize_text
from .search_utils import load_movies, CACHE_PATH



class InvertedIndex:
    def __init__(self):        
        self.index = defaultdict(set)   # Dictionary that maps tokens (strings) to sets of document IDs (integers)
        self.docmap = {}                # Dictionary that maps document IDs (int) to their full document object
        self.term_frequencies = {}      # Dictionary of document IDs to Counter objects



    def __add_document(self, doc_id, text):
        # Tokenise the input text
        tokens = tokenize_text(text)
        
        # Make sure this doc_id has a Counter
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        
        for token in tokens:
            # Add each token to the index with the document ID
            self.index[token].add(doc_id)
            # Increment the frequency counter for the term observed, in the specific document
            # The Counter wrapper handles instantiating at 0 and has some methods that might be helpful too            
            self.term_frequencies[doc_id][token] += 1       # Increment count
        
    
    
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

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        num_docs_with_term = len(self.index[token]) 
        num_docs_without_term = len(self.docmap) - num_docs_with_term
        bm25_idf = math.log((num_docs_without_term + 0.5) / (num_docs_with_term + 0.5) + 1)  # 0.5 and 1 are for edge cases and smoothing
        return bm25_idf

    def get_documents(self, token: str):
        # Get the set of document ID's for a given token (set it to lowercase)
        # Return them as a list, sorted in ascending order
        token = token.lower()
        return sorted(self.index.get(token, set()))
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        num_docs_with_term = len(self.index[token])
        return math.log((doc_count + 1) / (num_docs_with_term + 1))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]
    
    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def load(self):
        # Load the index and docmap from disk using pickle
        # File paths
        index_path = os.path.join(CACHE_PATH, "index.pkl")
        docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        termfreq_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        
        # Raise an error if the files don't exist
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"Docmap file not found: {docmap_path}")
        
        if not os.path.exists(termfreq_path):
            raise FileNotFoundError(f"Term frequencies file not found: {termfreq_path}")

        # Load index
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        # Load docmap
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        # Load term frequencies
        with open(termfreq_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        
    


    def save(self):
        # Save the index and docmap attributes to disk using the pickle modules dump function
        # Create the cache directory if it doesn't exist
        os.makedirs(CACHE_PATH, exist_ok=True)

        # File paths
        index_path = os.path.join(CACHE_PATH, "index.pkl")
        docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        termfreq_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")

        # Save index
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        # Save docmap
        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)        

        # Save term frequencies
        with open(termfreq_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)        



    def num_documents(self):
        # Number of documents in the index
        return len(self.docmap)
    
    def num_documents_with_token(self, token):
        # Number of documents that contain the specific token (term)
        return len(self.index.get(token, set()))   
    
    def num_unique_tokens(self):
        # Number of unique tokens (terms) in the index
        return len(self.index)
    
    
    def tokens_in_doc(self, doc_id):
        # Number of tokens (terms) in a specific document
        return sum(self.term_frequencies[doc_id].values())


    def total_token_usage(self, token):
        # Total number of times a given token (term) is found in every document
        return sum(
            counter.get(token, 0)
            for counter in self.term_frequencies.values()
        )
    
    def total_tokens(self):
        # Total number of tokens (terms) across all documents
        return sum(
            sum(counter.values())
            for counter in self.term_frequencies.values()
        )

    
