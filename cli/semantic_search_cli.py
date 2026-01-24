#!/usr/bin/env python3

import argparse
from lib.semantic_search import embed_query_text, embed_text, cmd_search, verify_model, verify_embeddings
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    
    #Add the verify command to the CLI interpreter
    subparsers.add_parser("verify", help="Verify the LLM model")

    # Remember when adding additional commands, that if we are only registering the command we don't need to set a variable equal to the subparsers.add_parser
    # variable = subparsers.add_parser is needed when we need subsequent lines to add parameters via .add_argument

    embed_query_parser = subparsers.add_parser("embedquery", help="Convert user search queries into embedding vectors")
    embed_query_parser.add_argument("query", type=str, help="embedquery query")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate the embedded values from text")
    embed_text_parser.add_argument("text", type=str, help="embed_text text")

    search_parser = subparsers.add_parser("search", help="Use semantic search to find movies by meaning")
    search_parser.add_argument("query", type=str, help="search query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help=f"Optionally limit the results (default: {DEFAULT_SEARCH_LIMIT})",)

    subparsers.add_parser("verify_embeddings", help="Verify the embedded values")
    









    args = parser.parse_args()

    match args.command:
        case "embedquery":
            embed_query_text(args.query)

        case "embed_text":
            embed_text(args.text)

        case "search":
            cmd_search(args.query, args.limit)

        case "verify":            
            verify_model()

        case "verify_embeddings":            
            verify_embeddings()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()