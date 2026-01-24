#import os
#import numpy as np

from .semantic_search import SemanticSearch
from .search_utils import load_movies



def cmd_chunk(text, chunk_size, overlap):    
    words = text.split()    
    chunks = []    
    
    n_words = len(words)
    i = 0
    
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap


    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")



def cmd_embed_query_text(query):
    ss = SemanticSearch()

    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cmd_embed_text(text):
    ss = SemanticSearch()

    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")



def cmd_search(query, limit):
    ss = SemanticSearch()
    
    docs = load_movies()
    ss.load_or_create_embeddings(docs)
    results = ss.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (score: {res['score']:.4f})\n   {res['description'][:100]}...")


def cmd_verify_embeddings():
    ss = SemanticSearch()

    #Load the documents from movies.json into a list.
    documents = load_movies()

    embeddings = ss.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def cmd_verify_model():
    ss = SemanticSearch()

    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
    