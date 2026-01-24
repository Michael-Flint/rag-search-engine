#!/usr/bin/env python3

import argparse
from lib.semantic_cmds import cmd_chunk, cmd_embed_query_text, cmd_embed_text, cmd_search, cmd_semantic_chunk,  cmd_verify_model, cmd_verify_embeddings
from lib.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_LIMIT, DEFAULT_CHUNK_OVERLAP, DEFAULT_SEMANTIC_CHUNK_SIZE

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    
    #Add the verify command to the CLI interpreter
    subparsers.add_parser("verify", help="Verify the LLM model")

    # Remember when adding additional commands, that if we are only registering the command we don't need to set a variable equal to the subparsers.add_parser
    # variable = subparsers.add_parser is needed when we need subsequent lines to add parameters via .add_argument

    chunk_parser = subparsers.add_parser("chunk", help="Implement fixed-size chunking to split long text for embedding")
    chunk_parser.add_argument("text", type=str, help="chunk text position")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_LIMIT, help=f"Optionally specify the chunk size (default: {DEFAULT_CHUNK_LIMIT})",)
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help=f"Optionally specify the chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})",)
    
    embed_query_parser = subparsers.add_parser("embedquery", help="Convert user search queries into embedding vectors")
    embed_query_parser.add_argument("query", type=str, help="embedquery query")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate the embedded values from text")
    embed_text_parser.add_argument("text", type=str, help="embed_text text")

    search_parser = subparsers.add_parser("search", help="Use semantic search to find movies by meaning")
    search_parser.add_argument("query", type=str, help="search query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help=f"Optionally limit the results (default: {DEFAULT_SEARCH_LIMIT})",)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Implement semantic based chunking to split long text for embedding")
    semantic_chunk_parser.add_argument("text", type=str, help="chunk text")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_SEMANTIC_CHUNK_SIZE, help=f"Optionally specify the chunk size (default: {DEFAULT_SEMANTIC_CHUNK_SIZE})",)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help=f"Optionally specify the chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})",)

    subparsers.add_parser("verify_embeddings", help="Verify the embedded values")
    









    args = parser.parse_args()

    match args.command:
        case "chunk":
            cmd_chunk(args.text, args.chunk_size, args.overlap)
        
        case "embedquery":
            cmd_embed_query_text(args.query)

        case "embed_text":
            cmd_embed_text(args.text)

        case "search":
            cmd_search(args.query, args.limit)

        case "semantic_chunk":
            cmd_semantic_chunk(args.text, args.max_chunk_size, args.overlap)

        case "verify":            
            cmd_verify_model()

        case "verify_embeddings":            
            cmd_verify_embeddings()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()