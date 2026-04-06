"""Standalone search script — runs in its own process, prints JSON to stdout.

Usage:
    python search_cli.py --query "a dog barking" --top-k 10 \
        --embeddings-dir /Volumes/Samsung_T7/dataset/embeddings

Called by demo.py via subprocess to keep torch completely out of the web server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from cross_modal.retrieval import CrossModalRetriever, DEFAULT_EMBEDDINGS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMBEDDINGS_DIR)
    args = parser.parse_args()

    retriever = CrossModalRetriever(
        embeddings_dir=args.embeddings_dir, device="cpu", use_fp16=False,
    )
    retriever.load_all()
    results = retriever.search(args.query, top_k=args.top_k)
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
