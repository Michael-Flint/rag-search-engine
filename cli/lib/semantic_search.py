import os
#import string
import numpy as np

from .search_utils import CACHE_PATH, load_movies
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):                
        self.document_map = {}
        self.documents = None
        self.embeddings = None
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_path = os.path.join(CACHE_PATH, "movie_embeddings.npy")

    def build_embeddings(self, documents):        
        self.documents = documents

        doc_string_rep = []

        # Documents is a list of dictionaries, each representing a movie
        for doc in documents:
            #Add each to self.document_map where the key is the ID and the value is the document
            self.document_map[doc["id"]] = doc
            #Add each to a list of string representations of the movies
            doc_string_rep.append(f"{doc['title']}: {doc['description']}")
    
        #Use the model to encode the movie strings
        embeddings = self.model.encode(doc_string_rep, show_progress_bar=True)
        self.embeddings = embeddings    

        # Save the embeddings         
        np.save(self.embeddings_path, embeddings)

        return self.embeddings
    


    def generate_embedding(self, text):        
        if len(text.strip()) == 0:
            raise ValueError("text cannot be empty or white space")
        
        input_list = []  #The encode method expects a list of inputs

        #For now, we only have a single term but add it to the list to allow .encode to operate
        input_list.append(text)

        return self.model.encode(input_list)[0]



    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        # Raise an error if the files don't exist
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)           
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        # Length does not match, rebuild
        return self.build_embeddings(documents)
    


def embed_query_text(query):
    ss = SemanticSearch()

    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def embed_text(text):
    ss = SemanticSearch()

    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")



def verify_embeddings():
    ss = SemanticSearch()

    #Load the documents from movies.json into a list.
    documents = load_movies()

    embeddings = ss.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def verify_model():
    ss = SemanticSearch()

    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
    