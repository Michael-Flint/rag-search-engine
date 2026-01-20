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


    def get_documents(self, token: str):
        # Get the set of document ID's for a given token (set it to lowercase)
        # Return them as a list, sorted in ascending order
        token = token.lower()
        return sorted(self.index.get(token, set()))
    


    def get_tf(self, doc_id, term):
        tokenised_term = tokenize_text(term)
        # term should be a single token, otherwise raise exception
        if len(tokenised_term) != 1:
            #print(f"Term: {term}, tokenised: {tokenised_term}")
            raise Exception("Term should be a single word")

        single_term = tokenised_term[0]
        
        #if tokenised_term not in self.term_frequencies[doc_id]:
        if doc_id not in self.term_frequencies:
            # If term not found in the dictionary, return 0
            return 0
        # Otherwise, return the # times it is found in the specific document  
        # Counter automatically returns a 0 if the term is not found
        return self.term_frequencies[doc_id][single_term]

    
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

