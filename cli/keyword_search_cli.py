#!/usr/bin/env python3

import argparse
from pathlib import Path
from file_retriever import load_movies
from text_cleaner import cleaner


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies_list = load_movies()
            matches = []
            query_words = []
            query_words = cleaner(args.query).split()
            title_words = []

            for movie in movies_list:                
                title_words = cleaner(movie.get("title", "")).split()  
                # Split without arguments automatically strips whitespace and ignores empty tokens

                #ChatGPT version:  if any(qw in tw for qw in query_words for tw in title_words):

                match_found = False                
                for qw in query_words:
                    for tw in title_words:
                        if qw in tw:
                            match_found = True
                            break  # stop checking more title words for this query word
                    if match_found:
                        break      # stop checking other query words

                if match_found:
                    matches.append(movie)

            
            if matches:
                top_matches = sorted(matches, key=lambda m: m["id"])[:5]                
                for idx, movie in enumerate(top_matches, start=1):
                    print(f"{idx}. {movie['title']}")
            else:
                print("No matches found.")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()