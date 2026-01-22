#!/usr/bin/env python3

import argparse
import sys

from lib.keyword_search import search_command, tokenize_text
from lib.index import InvertedIndex
from lib.search_utils import DEFAULT_SEARCH_LIMIT, BM25_K1, BM25_B

def search_and_print(idx, tokens):
    results = []
    seen_ids = set()

    for token in tokens:
        # Get matching document IDs for this token
        doc_ids = idx.get_documents(token)

        for doc_id in doc_ids:
            # Avoid duplicates
            if doc_id in seen_ids:
                continue

            seen_ids.add(doc_id)
            results.append(doc_id)

            # Stop once we have 5 results
            if len(results) == 5:
                break

        if len(results) == 5:
            break


    # Print the resulting document titles and IDs.
    for doc_id in results:
        movie = idx.docmap[doc_id]
        print(f"{movie['title']} (ID: {doc_id})")
        #    print(f"{i}. {res['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    #Search command
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")    #Search = verb, "Do a search"
    search_parser.add_argument("query", type=str, help="Search query")                  #Query = noun,  "I want you to search for NOUN"

    #Inverse document frequency (rare terms are more important than common ones)
    idf_parser = subparsers.add_parser("idf", help="Test the rarity of a term")
    idf_parser.add_argument("term", type=str, help="Term to be tested")

    #Build an index 
    build_parser = subparsers.add_parser("build", help="Build movie index and save it to disk")

    #Index, regardless
    index_parser = subparsers.add_parser("index", help="Rebuild the index")
    index_parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuilding even if index exists",
    )

    #TermFrequency command
    tf_parser = subparsers.add_parser("tf", help="Count the instances of a term in a DocumentID")
    tf_parser.add_argument("doc_id", type=int, help="tf DocumentID term")
    tf_parser.add_argument("term", type=str, help="tf DocumentID term")


    #TermFrequencyInverseDocFrequency command
    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate the TermFrequency-InverseDocumentFrequency of a term")
    tfidf_parser.add_argument("doc_id", type=int, help="tfidf DocumentID term")
    tfidf_parser.add_argument("term", type=str, help="tfidf DocumentID term")


    #Okapi BM25 version of IDF
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Okapi BM25 version of Inverse Document Frequency")
    bm25_idf_parser.add_argument("term", type=str, help="bm25idf term")

    #Okapi BM25 version of TF
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Okapi BM25 version of Term Frequency")
    bm25_tf_parser.add_argument("doc_id", type=int, help="bm25tf DocumentID term")
    bm25_tf_parser.add_argument("term", type=str, help="bm25tf DocumentID term")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 k1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    #Okapi BM25 search parser
    bm25_search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help=f"Optionally limit the results (default: {DEFAULT_SEARCH_LIMIT})",)


    args = parser.parse_args()

    match args.command:
        case "bm25idf":
            idx = InvertedIndex()

            try:            
                idx.load()
            except FileNotFoundError as e:
                print("Index not found, run build first")
                sys.exit(1)
            bm25idf = idx.get_bm25_idf(args.term)            
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        
        case "bm25tf":
            idx = InvertedIndex()

            try:            
                idx.load()
            except FileNotFoundError as e:
                print("Index not found, run build first")
                sys.exit(1)
            bm25tf = idx.get_bm25_tf(args.doc_id, args.term)            
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":            
            print("Loading index")
            idx = InvertedIndex()
            try:            
                idx.load()
            except FileNotFoundError as e:
                print("Index not found, run build first")
                sys.exit(1)
            print(f"Searching for: {args.query}")

            search_results = idx.bm25_search(args.query, args.limit)

            for i, res in enumerate(search_results, 1):
                print(f"{i}. ({res['doc_id']}) {res['title']} - Score: {res['score']:.2f}")

            

        
        
        case "build":
            print("Building movie index")
            # Instantiate the class
            idx = InvertedIndex()
            # Build the index
            idx.build()
            # Save it to disk
            idx.save()            

        case "idf":
            idx = InvertedIndex()

            try:            
                idx.load()
            except FileNotFoundError as e:
                print("Index not found, run build first")
                sys.exit(1)
            idf = idx.get_idf(args.term)            
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "index":
            print(f"Rebuilding index, force={args.force}")
            
        case "search":            
            print("Loading index")
            idx = InvertedIndex()
            try:            
                idx.load()
            except FileNotFoundError as e:
                print("Index not found, run build first")
                sys.exit(1)
            print(f"Searching for: {args.query}")

            # Iterate over each token in the query and use the index to get any matching documents for each token.            
            search_and_print(idx, tokenize_text(args.query))

        case "tf": 
            #print("Loading index")
            idx = InvertedIndex()

            try:            
                idx.load()
            except FileNotFoundError as e:
                print("Index not found, run build first")
                sys.exit(1)

            #print(f"Searching documentID: {args.doc_id} for: {args.term}")
            
            term_freq = idx.get_tf(args.doc_id, args.term)
            #print(f"Found {term_freq} instances of {args.term} in document_id: {args.doc_id}")
            print(term_freq)

        case "tfidf": 
            #print("Loading index")
            idx = InvertedIndex()

            try:            
                idx.load()
            except FileNotFoundError as e:
                print("Index not found, run build first")
                sys.exit(1)
            
            tf_idf = idx.get_tf_idf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()


