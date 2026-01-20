#!/usr/bin/env python3

import argparse
import sys

from lib.keyword_search import search_command, tokenize_text
from lib.index import InvertedIndex


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
    search_parser = subparsers.add_parser("tf", help="Count the instances of a term in a DocumentID")
    search_parser.add_argument("doc_id", type=int, help="tf DocumentID term")
    search_parser.add_argument("term", type=str, help="tf DocumentID term")



    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building movie index")
            # Instantiate the class
            idx = InvertedIndex()
            # Build the index
            idx.build()
            # Save it to disk
            idx.save()            

        case "index":
            print(f"Rebuilding index, force={args.force}")
            pass

        case "search":            
            print("Loading index")
            idx = InvertedIndex()

            try:            
                idx.load()
            except FileNotFoundError as e:
                print("Index not found, run build first")
                sys.exit(1)

            print(f"Searching for: {args.query}")

            #results = search_command(args.query)
            #for i, res in enumerate(results, 1):
            #    print(f"{i}. {res['title']}")

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


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()


