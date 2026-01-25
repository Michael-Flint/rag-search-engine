import json
import numpy as np
import os
import re


from .search_utils import CACHE_PATH, load_movies, format_search_result, DOCUMENT_PREVIEW_LENGTH
from sentence_transformers import SentenceTransformer
from typing import List

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.document_map = {}
        self.documents = None                               # Documents is a list of dictionaries, each representing a movie
        self.embeddings = None
        self.embeddings_path = os.path.join(CACHE_PATH, "movie_embeddings.npy")
        self.embeddings_index_path = os.path.join(CACHE_PATH, "movie_embeddings_id_map.npy")
        self.id_to_index = {}                               # doc_id -> row index in embeddings        
        self.model = SentenceTransformer(model_name)        # Load the model (downloads automatically the first time)


    def build_embeddings(self, documents):        
        self.documents = documents

        doc_string_rep = []

        
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
        #for doc in self.documents:
        #    doc_id = doc["id"]
        #    doc_vec = self.get_vector_for_id(doc_id)
        #    score = cosine_similarity(query_embedding, doc_vec)
        #    #Create a list of (similarity_score, document) tuples
        #    similarity_list.append(
        #        {
        #            "score": float(score),
        #            "title": doc["title"],
        #            "description": doc["description"],
        #        }
        #    )
        #Sort by similarity, descending
        #similarity_list.sort(key=lambda x: x["score"], reverse=True)
        #Return top `limit` results
        #return similarity_list[:limit]

        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarity_list.append((similarity, self.documents[i]))

        #Sort by similarity, descending
        similarity_list.sort(key=lambda x: x["score"], reverse=True)

        results = []
        for score, doc in similarity_list[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results
        
    
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
    

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name = model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_PATH, "chunk_metadata.json")


    def build_chunk_embeddings(self, documents):        
        self.documents = documents

        doc_string_rep = []
        all_chunks = []     #list of all chunk strings across all documents
        chunk_metadata = [] #list of dictionaries
        

        # Documents is a list of dictionaries, each representing a movie
        for movie_idx, doc in enumerate(documents):
            description = doc["description"]
            #If description is empty, skip this movie
            if not description.strip():
                continue

            #4 sentence chunks, with 1 sentence overlap
            chunks = semantic_chunk(description, 4, 1)

            #For each chunk, add a dictionary to the chunk metadata list with:
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks) #Total number of chunks in the document
                })
                
            
        #Use the model to encode the chunks                
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        
        #Save the chunk metadata 
        self.chunk_metadata = chunk_metadata

        # save both embeddings and the id map
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings
    

    def search_chunks(self, query: str, limit: int=10):
        #Generate an embedding of the query using the method from SemanticSearch
        query_embedding = self.generate_embedding(query)
        
        #Populate an empty list to store chunk_score dictionaries
        chunk_scores = []
        
        #For each chunk embedding
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            #Calculate the cosine similarity between the chunk embedding and the query embedding            
            similarity = cosine_similarity(query_embedding, chunk_embedding)

            chunk_metadata = self.chunk_metadata[i]

            #Append a dictionary to the chunk score list with fields
            chunk_scores.append({
                "chunk_idx": chunk_metadata["chunk_idx"],
                "movie_idx": chunk_metadata["movie_idx"],
                "score": similarity         #Cosine similarity score
             })

        #Create an empty dictionary that maps movie indexes to their scores
        movie_scores = {}

        #For each chunk score 
        for chunk_score in chunk_scores:
            #if the movie-idx is not in the movie score dictionary yet, or the new score is higher than the existing one
            #update the movie score dictionary with the new chunk score
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]

            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score

        #Sort the movie scores by score, descending
        sorted_movies = sorted(
            movie_scores.items(),  # gives (movie_idx, score)
            key=lambda item: item[1],  # sort by score
            reverse=True,
        )
        
        
        results = []
        for movie_idx, score in sorted_movies[:limit]:
            doc = self.documents[movie_idx]
            metadata = {}

            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )


        return results 

        
    


    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            #Add each to self.document_map where the key is the ID and the value is the document
            self.document_map[doc["id"]] = doc
        
        #If the embeddings and metadata already exist, return them
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)           
            with open(self.chunk_metadata_path, "r") as f:            
                metadata_json = json.load(f)
            
            self.chunk_metadata = metadata_json["chunks"]

            return self.chunk_embeddings

        #Files don't exist, rebuild the chunk embeddings and metadata
        return self.build_chunk_embeddings(documents)



def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_index_by_field(data_list, field_to_search, search_term):    
    for index, item in enumerate(data_list):
        if item[field_to_search] == search_term:
            return index    
    
    
def semantic_chunk(text, max_chunk_size, overlap) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    return chunks