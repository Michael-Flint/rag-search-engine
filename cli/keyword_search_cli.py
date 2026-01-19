#!/usr/bin/env python3

import argparse
from pathlib import Path
from retriever import load_movies


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

            for movie in movies_list:
                title = movie.get("title", "")
                if args.query in title:
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