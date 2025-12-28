#!/usr/bin/env python3
"""
CLI script to embed a JSONL dataset using span-based embeddings
and save embeddings to an HDF5 store.

Dependencies:
    - torch
    - tqdm
    - h5py
"""

import argparse
from tqdm import tqdm

from src.span_embedder import BertSpanEmbedder, DecoderSpanEmbedder
from src.dataset import load_jsonl
from src.embedding_store import EmbeddingStore


def main():
    parser = argparse.ArgumentParser(description="Embed a JSONL dataset and store embeddings")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL dataset")
    parser.add_argument("--out-h5", type=str, required=True, help="Path to output HDF5 embedding store")
    parser.add_argument("--model", type=str, required=True, help="HF model name or local path")
    parser.add_argument("--model-type", choices=["bert", "decoder"], default="bert", help="Model type")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--layer-pool", choices=[None, "last", "last4mean"], default=None, help="Layer pooling strategy")
    parser.add_argument("--subtoken-pool", choices=["mean", "max"], default="mean", help="Subtoken pooling strategy")
    parser.add_argument("--lexem-filter", type=str, default=None, help="Only embed samples with this lexem")
    args = parser.parse_args()

    # ---------------------------------------------------
    # Initialize embedder
    # ---------------------------------------------------
    if args.model_type == "bert":
        embedder = BertSpanEmbedder(args.model, device=args.device)
    else:
        embedder = DecoderSpanEmbedder(args.model, device=args.device)

    # ---------------------------------------------------
    # Load dataset (optionally filtered)
    # ---------------------------------------------------
    dataset_iter = load_jsonl(args.input, lexem_filter=args.lexem_filter)

    # ---------------------------------------------------
    # Open embedding store
    # ---------------------------------------------------
    with EmbeddingStore(args.out_h5, mode="a") as store:
        for sample in tqdm(dataset_iter, desc="Embedding samples"):
            sample_id = sample["id"]

            if sample_id in store.all_ids():
                continue  # resume-safe

            vec = embedder.encode(
                sentence=sample["sentence"],
                span=tuple(sample["span"]),
                layer_pool=args.layer_pool,
                subtoken_pool=args.subtoken_pool
            )

            store.save({sample_id: vec})


if __name__ == "__main__":
    main()
