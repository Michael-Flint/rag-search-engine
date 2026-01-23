#!/usr/bin/env python3

import argparse
from lib.semantic_search import embed_text, verify_model

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    
    #Add the verify command to the CLI interpreter
    subparsers.add_parser("verify", help="Verify the LLM model")

    # Remember when adding additional commands, that if we are only registering the command we don't need to set a variable equal to the subparsers.add_parser
    # variable = subparsers.add_parser is needed when we need subsequent lines to add parameters via .add_argument


    embed_text_parser = subparsers.add_parser("embed_text", help="Generate the embedded values from text")
    embed_text_parser.add_argument("text", type=str, help="embed_text text")


    args = parser.parse_args()

    match args.command:
        case "embed_text":
            embed_text(args.text)

        case "verify":            
            verify_model()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()