"""
build_protein_features.py

Parses the node2vec embedding CSV (node2vec_with_uniprot.csv) and generates:
  - protein_features_ppi.npy  : (N, 128) float32 array of node2vec embeddings
  - protein_ids_ppi.json      : list of N UniProt IDs corresponding to each row

Usage:
    python build_protein_features.py \
        --input  /path/to/node2vec_with_uniprot.csv \
        --out_npy /path/to/protein_features_ppi.npy \
        --out_ids /path/to/protein_ids_ppi.json

Defaults match the UConn Health HPC paths used during development.
"""

import argparse
import json
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# 1. Argument parsing
# ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Build protein PPI feature matrix")
    parser.add_argument(
        "--input",
        default="/home/FCAM/juli/HRP/notebooks/node2vec_with_uniprot.csv",
        help="Path to node2vec_with_uniprot.csv"
    )
    parser.add_argument(
        "--out_npy",
        default="/home/FCAM/juli/HRP/notebooks/protein_features_ppi.npy",
        help="Output path for the .npy embedding matrix"
    )
    parser.add_argument(
        "--out_ids",
        default="/home/FCAM/juli/HRP/notebooks/protein_ids_ppi.json",
        help="Output path for the protein ID list (.json)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
# 2. Vector parser
#    The 'value' column is stored as a stringified list, e.g.:
#    "['0.246' '-0.111' ... '0.025\\n']"
#    We strip brackets, quotes, and newline escapes, then split on whitespace.
# ─────────────────────────────────────────────────────────────────
def parse_vec(s: str) -> np.ndarray:
    cleaned = (
        s.strip()
         .strip("[]")
         .replace("'", "")
         .replace("\\n", " ")
         .replace("\n",  " ")
    )
    return np.array(cleaned.split(), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)

    print(f"  Rows: {len(df)}, unique UniProt IDs: {df['UniProt_ID'].nunique()}")

    # Parse all embedding vectors
    print("Parsing node2vec vectors...")
    vecs = np.stack(df["value"].apply(parse_vec).values)   # (N, 128)
    ids  = df["UniProt_ID"].tolist()

    print(f"  Embedding matrix shape : {vecs.shape}")
    print(f"  dtype                  : {vecs.dtype}")
    print(f"  value range            : [{vecs.min():.4f}, {vecs.max():.4f}]")
    print(f"  NaN present            : {np.isnan(vecs).any()}")
    print(f"  Inf present            : {np.isinf(vecs).any()}")

    # Save
    np.save(args.out_npy, vecs)
    with open(args.out_ids, "w") as f:
        json.dump(ids, f)

    print(f"\nSaved embedding matrix → {args.out_npy}")
    print(f"Saved protein ID list  → {args.out_ids}")


if __name__ == "__main__":
    main()
