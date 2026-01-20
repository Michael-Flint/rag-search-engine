#!/usr/bin/env python3

import argparse

from lib.keyword_search import search_command
from lib.index import InvertedIndex


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

            # Print a message containing the first ID of the document for the token 'merida', which should be document 4651                            
            docs = idx.get_documents("merida")
            if docs:
                print(f"First document ID for token 'merida' is {docs[0]}")
            else:
                print("No documents found for token 'merida'")

        case "index":
            print(f"Rebuilding index, force={args.force}")
            pass

        case "search":            
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
