import os
import numpy as np

from .search_utils import CACHE_PATH, load_movies
from .search_utils import CACHE_PATH, load_movies
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):                
        self.document_map = {}
        self.documents = None
        self.embeddings = None
        self.embeddings_path = os.path.join(CACHE_PATH, "movie_embeddings.npy")
        self.embeddings_index_path = os.path.join(CACHE_PATH, "movie_embeddings_id_map.npy")
        self.id_to_index = {}    # doc_id -> row index in embeddings        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')    # Load the model (downloads automatically the first time)
        

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

        # map each doc_id to its row index in the embeddings array
        self.id_to_index = {doc["id"]: i for i, doc in enumerate(documents)}

        # save both embeddings and the id map
        np.save(self.embeddings_path, embeddings)
        np.save(self.embeddings_index_path, self.id_to_index)
   
        return self.embeddings
    


    def generate_embedding(self, text):        
        if len(text.strip()) == 0:
            raise ValueError("text cannot be empty or white space")
        
        input_list = []  #The encode method expects a list of inputs

        #For now, we only have a single term but add it to the list to allow .encode to operate
        input_list.append(text)

        return self.model.encode(input_list)[0]
    
    def get_vector_for_id(self, doc_id):
        idx = self.id_to_index[doc_id]
        return self.embeddings[idx]

   

    def search(self, query, limit):
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        #Generate embedding 
        query_embedding = self.generate_embedding(query)

        similarity_list = []

        #Calculate cosine similarity between the query embedding and each document embedding
        for doc in self.documents:
            doc_id = doc["id"]
            doc_vec = self.get_vector_for_id(doc_id)
            score = cosine_similarity(query_embedding, doc_vec)

            #Create a list of (similarity_score, document) tuples
            similarity_list.append(
                {
                    "score": float(score),
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        #Sort by similarity, descending
        similarity_list.sort(key=lambda x: x["score"], reverse=True)

        #Return top `limit` results
        return similarity_list[:limit]
        
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        # Raise an error if the files don't exist
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)           
            if len(self.embeddings) == len(documents):
                self.id_to_index = np.load(self.embeddings_index_path, allow_pickle=True).item()
                return self.embeddings
        
        # Length does not match, rebuild
        return self.build_embeddings(documents)
    


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
