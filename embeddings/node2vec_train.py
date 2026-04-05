"""
node2vec_train.py

Trains node2vec embeddings on the STRING v12.0 physical interaction network.
Performs a grid search over (p, q) hyperparameters and saves the best model.

Usage:
    python node2vec_train.py \
        --input  /path/to/9606.protein.physical.links.full.v12.0.txt.gz \
        --outdir /path/to/results/ \
        --cutoff 200

Defaults match the UConn Health HPC paths used during development.
Requires: node2vec, networkx, pandas, numpy
"""

import argparse
import gzip
import os

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# 1. Argument parsing
# ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train node2vec on STRING PPI network")
    parser.add_argument(
        "--input",
        default="/home/FCAM/juli/HRP/data/9606.protein.physical.links.full.v12.0.txt.gz",
        help="Path to STRING physical links file (.txt.gz)"
    )
    parser.add_argument(
        "--outdir",
        default="/home/FCAM/juli/HRP/results/",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=200,
        help="Minimum combined_score to keep an edge (default: 200)"
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=128,
        help="Embedding dimensions (default: 128)"
    )
    parser.add_argument(
        "--walk_length",
        type=int,
        default=80,
        help="Random walk length (default: 80)"
    )
    parser.add_argument(
        "--num_walks",
        type=int,
        default=20,
        help="Number of walks per node (default: 20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
# 2. Load STRING network
# ─────────────────────────────────────────────────────────────────
def load_network(data_file: str, cutoff: int) -> nx.Graph:
    print(f"Loading STRING network from: {data_file}")
    print(f"  Edge score cutoff: >= {cutoff}")

    chunks = []
    with gzip.open(data_file, "rt") as fh:
        reader = pd.read_csv(
            fh,
            sep=r"\s+",
            comment="#",
            header=0,
            usecols=["protein1", "protein2", "combined_score"],
            dtype={
                "protein1":       "category",
                "protein2":       "category",
                "combined_score": "int32",
            },
            chunksize=2_000_000,
            low_memory=False,
        )
        for chunk in tqdm(reader, desc="  Reading chunks"):
            chunk = chunk[chunk.combined_score >= cutoff]
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Kept {len(df):,} edges >= {cutoff}")

    G = nx.from_pandas_edgelist(
        df, "protein1", "protein2",
        edge_attr="combined_score",
        create_using=nx.Graph
    )
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ─────────────────────────────────────────────────────────────────
# 3. Grid search over (p, q)
# ─────────────────────────────────────────────────────────────────
def mean_similarity(model, pairs):
    sims = [model.wv.similarity(u, v) for u, v in pairs]
    return float(np.mean(sims))


def grid_search(G: nx.Graph, dims: int, walk_length: int,
                num_walks: int, seed: int, tmp_dir: str):

    GRID_PQ = [(1, 0.5), (1, 1), (4, 1), (4, 4)]

    # sample up to 10k edges for evaluation
    edges_list = list(G.edges())
    sample_pairs = edges_list[:10_000] if len(edges_list) >= 10_000 else edges_list

    best_model, best_score, best_pq = None, -np.inf, None

    for p, q in GRID_PQ:
        print(f"\n  Training node2vec: p={p}, q={q}")
        n2v = Node2Vec(
            G,
            dimensions=dims,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p, q=q,
            weight_key="combined_score",
            workers=os.cpu_count(),
            temp_folder=tmp_dir,
            seed=seed,
            quiet=True,
        )
        model = n2v.fit(window=10, min_count=1, batch_words=2048, epochs=1)
        score = mean_similarity(model, sample_pairs)
        print(f"  Mean edge similarity: {score:.4f}")

        if score > best_score:
            best_model, best_score, best_pq = model, score, (p, q)

    print(f"\n  Best (p, q) = {best_pq}  |  mean edge similarity = {best_score:.4f}")
    return best_model, best_pq


# ─────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tmp_dir = os.path.join(args.outdir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Load network
    G = load_network(args.input, args.cutoff)

    # Grid search
    print("\nRunning (p, q) grid search...")
    best_model, best_pq = grid_search(
        G,
        dims=args.dims,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        seed=args.seed,
        tmp_dir=tmp_dir,
    )

    # Save
    txt_path = os.path.join(args.outdir, "ppi_node2vec.emb.defult.txt")
    bin_path = os.path.join(args.outdir, "ppi_node2vec.kv")

    best_model.wv.save_word2vec_format(txt_path)
    best_model.wv.save(bin_path)

    print(f"\nSaved text embeddings → {txt_path}")
    print(f"Saved binary keyedvectors → {bin_path}")
    print(f"Best hyperparameters: p={best_pq[0]}, q={best_pq[1]}")

    # Quick sanity check
    node0 = best_model.wv.index_to_key[0]
    print(f"Sample vector ({node0})[:5] = {best_model.wv[node0][:5]}")


if __name__ == "__main__":
    main()
