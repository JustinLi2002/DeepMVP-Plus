# ContextPTM

**ContextPTM: Context-Aware PTM Site Prediction via Protein Interaction Network Embeddings and Kinase Priors**

*Justin Li, Ji Yu · Center for Cell Analysis and Modeling, UConn Health*

---

## Overview

DeepMVP (Wen et al., *Nature Methods* 2025) predicts PTM sites using only local sequence windows, ignoring protein-level context. This repository proposes **ContextPTM**, which augments the CNN+BiGRU ensemble with two complementary protein-level feature sets:

1. **node2vec PPI embeddings (128-dim)**: learned from the STRING v12.0 physical interaction network via node2vec
2. **Kinase prediction scores (605-dim)**: aggregated per-protein scores from GPS 5.0, NetworkKIN, and PhosphoPICK
3. **Combined (733-dim)**: both feature types concatenated

All variants are evaluated against a reproduced DeepMVP baseline across all 8 PTM types.

---

## Results

All AUROC values on held-out test sets. Ensemble of 10 models with IQR outlier-excluded averaging.

| PTM | Original DeepMVP | Baseline | +Kinase | +node2vec PPI | +Combined |
|-----|------------------|----------|---------|---------------|-----------|
| Phosphorylation S/T | 0.950 | 0.9510 | 0.9665 | 0.9654 | **0.9672** |
| Phosphorylation Y | 0.910 | 0.8710 | 0.9321 | 0.9220 | **0.9340** |
| Sumoylation K | 0.850 | 0.8615 | 0.9256 | 0.9210 | **0.9269** |
| Ubiquitination K | 0.870 | 0.8814 | 0.9296 | 0.9248 | **0.9307** |
| Acetylation K | 0.900 | 0.9049 | 0.9615 | 0.9560 | **0.9623** |
| N-Glycosylation N | 0.980 | 0.9868 | 0.9968 | 0.9954 | **0.9969** |
| Methylation K | 0.950 | 0.9503 | 0.9814 | 0.9785 | **0.9815** |
| Methylation R | 0.960 | 0.9306 | 0.9810 | 0.9738 | **0.9811** |

All three feature variants consistently outperform the original DeepMVP across all 8 PTM types. The combined model achieves the best performance on all PTM types, suggesting that PPI network topology and kinase regulatory context provide complementary information.

---

## Repository Structure

```
ContextPTM/
│
├── README.md
│
├── models/                         # PTM prediction models
│   ├── deepmvp_reproduce_v2.py     # Baseline: CNN+BiGRU ensemble, sequence only
│   ├── deepmvp_ppi.py              # +node2vec PPI embeddings (128-dim)
│   ├── deepmvp_kinase.py           # +Kinase prediction scores (605-dim)
│   └── deepmvp_combined.py         # +Combined (node2vec + kinase, 733-dim)
│
├── embeddings/                     # Feature generation scripts
│   ├── node2vec_train.py           # Train node2vec on STRING PPI network
│   └── build_protein_features.py   # Parse node2vec output → protein_features_ppi.npy
│
├── slurm/                          # SLURM job submission scripts
│   ├── submit_deepmvp_noppi.sh
│   ├── submit_deepmvp_ppi.sh
│   ├── submit_deepmvp_ppi_n2v.sh
│   └── submit_deepmvp_kinase.sh
│
└── notebooks/                      # Exploratory analysis (poster work)
    └── exploratory/
```

---

## Requirements

```bash
pip install torch numpy pandas scikit-learn tqdm node2vec
```

Tested with Python 3.10, PyTorch 2.x, CUDA 12.8 (NVIDIA A100).

---

## Data

Training and test data are from [PTMAtlas](http://deepmvp.ptmax.org) (Wen et al. 2025), the same dataset used to train the original DeepMVP. Files follow the format:

```
protein  aa  pos  x  y
```

where `x` is a 31-character amino acid window centered on the candidate site and `y` is the binary label.

PPI network: [STRING v12.0](https://string-db.org/) physical interaction edges, filtered at `combined_score ≥ 200`.

---

## Reproducing Results

### Step 1: Generate node2vec PPI embeddings

```bash
# Train node2vec on STRING network
python embeddings/node2vec_train.py

# Parse output and map to UniProt IDs
python embeddings/build_protein_features.py
```

This produces:
- `protein_features_ppi.npy` — shape (18423, 128)
- `protein_ids_ppi.json` — corresponding UniProt IDs

### Step 2: Run baseline (sequence only)

```bash
sbatch slurm/submit_deepmvp_noppi.sh
```

Or directly:
```bash
python models/deepmvp_reproduce_v2.py
```

### Step 3: Run +node2vec PPI

```bash
sbatch slurm/submit_deepmvp_ppi_n2v.sh
```

Or directly:
```bash
python models/deepmvp_ppi_n2v.py
```

### Step 4: Run +kinase scores

Requires `protein_features.npy` and `protein_ids.json` (kinase prediction scores from GPS/NetworkKIN/PhosphoPICK, generated separately).

```bash
sbatch slurm/submit_deepmvp_ppi.sh
```

---

## Model Architecture

The baseline follows the CNN+BiGRU architecture described in DeepMVP (Extended Data Fig. 3c), re-implemented in PyTorch:

```
Input: one-hot encoded sequence window (23 × 31)
  → Conv1D(512, k=5) + BN + LeakyReLU + Dropout  ×3
  → Bidirectional GRU (50 units) → hidden state concat → (100,)
  → FC(100→64) + BN + LeakyReLU + Dropout
  → FC(64→1)
```

The +PPI variant adds a parallel branch:
```
PPI vector (128-dim or 605-dim)
  → FC(dim→128) + LeakyReLU → (128,)
  
Concat([sequence_branch(100,), ppi_branch(128,)]) → (228,)
  → FC(228→64) + BN + LeakyReLU + Dropout
  → FC(64→1)
```

Ten models are trained per PTM type (different random seeds), with early stopping on validation AUROC. Final predictions use IQR outlier-excluded averaging across the 10 models.

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping patience | 10 |
| Val split | 10% of train |
| Loss | BCEWithLogitsLoss + pos_weight |
| Ensemble size | 10 models |

---

## Citation

If you use this work, please also cite the original DeepMVP paper:

```
Wen B, Wang C, Li K, et al. DeepMVP: deep learning models trained on
high-quality data accurately predict PTM sites and variant-induced alterations.
Nature Methods, 2025.
```

