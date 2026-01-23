#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    
    #Add the verify command to the CLI interpreter
    subparsers.add_parser("verify", help="Verify the LLM model")

    # Remember when adding additional commands, that if we are only registering the command we don't need to set a variable equal to the subparsers.add_parser
    # variable = subparsers.add_parser is needed when we need subsequent lines to add parameters via .add_argument




    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()