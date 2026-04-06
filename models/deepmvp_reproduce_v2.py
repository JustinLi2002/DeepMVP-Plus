"""
DeepMVP Original Reproduction Script v2 (No PPI)
Architecture Reference: Extended Data Fig. 3c
- Multi-layer CNN (LeakyReLU + BatchNorm + Dropout)
- Fixed Bidirectional GRU (50 units)
- Dense Layer
- 10-Model Ensemble: IQR Outlier-Excluded Average
"""

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import copy
import os


DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_MODELS = 10         
MAX_EPOCHS = 100    
PATIENCE   = 10         
BATCH_TRAIN = 64       
BATCH_TEST  = 512
LR          = 1e-3

AA_LIST  = list("ACDEFGHIKLMNPQRSTVWY") + ["U", "O", "X"]
aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}
C        = len(AA_LIST)  # 23
EYE      = np.eye(C, dtype=np.float32)

#1. Dataset
class PTMDataset(Dataset):
    """Original Input: Sequence Only, No PPI"""
    def __init__(self, df):
        self.seq = df["x"].values
        self.y   = df["y"].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        s     = self.seq[i]
        idx   = [aa_to_idx[a] for a in s]
        x_seq = EYE[idx].T.astype(np.float32)   # (23, L)
        return torch.from_numpy(x_seq), torch.tensor(self.y[i])


# ─────────────────────────────────────────
# 2. Model Architecture(Refer to Extended Data Fig. 3c)
# ─────────────────────────────────────────
class DeepMVP_Single(nn.Module):
"""
CNN + Bidirectional GRU + Dense
Designed to align as closely as possible with Fig. 3c of the original paper:
conv1: in_ch → 512, kernel=5, BN, LeakyReLU, Dropout
conv2: 512 → 512,   kernel=5, BN, LeakyReLU, Dropout
conv3: 512 → 128,   kernel=5, BN, activation, Dropout
BiGRU: 128 → 50*2=100
Flatten → Dense(5700 → 64) → BN → activation → Dropout
Dense(64 → 1)
Note: 5700 = 100 * seq_len; specifically, when seq_len=31, this value is 3100. AdaptiveAvgPool is used here for normalization.
In reality, the original paper extracts the final hidden state from the GRU rather than flattening the outputs across all time steps.
Therefore, the *last* hidden state is used here: (batch, 100).
"""
    def __init__(self, in_ch=23, seq_len=31, dropout=0.3):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv1d(in_ch, 512, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 128, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(128)
        self.drop  = nn.Dropout(dropout)

        # GRU，50 units
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=50,
            batch_first=True,
            bidirectional=True
        ) 

        # Dense
        self.fc1    = nn.Linear(100, 64)
        self.bn4    = nn.BatchNorm1d(64)
        self.drop2  = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, 23, L)
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.drop(F.leaky_relu(self.bn3(self.conv3(x))))

        # CNN: (batch, 128, L) → GRU (batch, L, 128)
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)           # h: (2, batch, 50)
        x = torch.cat([h[0], h[1]], dim=1)  # (batch, 100)

        x = self.drop2(F.leaky_relu(self.bn4(self.fc1(x))))
        return self.fc_out(x).squeeze(-1)



def train_one_model(train_loader, val_loader, seed, device,
                    max_epochs=MAX_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = DeepMVP_Single().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # pos_weight
    all_y = np.concatenate([yb.numpy() for _, yb in train_loader.dataset
                             .__class__(train_loader.dataset.seq,
                                        train_loader.dataset.y)
                             ]) if False else None
    ys        = train_loader.dataset.y
    neg, pos  = (ys == 0).sum(), (ys == 1).sum()
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc  = 0.0
    best_state    = None
    wait          = 0

    for epoch in range(1, max_epochs + 1):
        # ── train ──
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # ── validate ──
        val_auc, _ = eval_loader_single(val_loader, model, device)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = copy.deepcopy(model.state_dict())
            wait         = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [seed={seed}] Early stop @ epoch {epoch}, best val AUC={best_val_auc:.4f}")
                break

    model.load_state_dict(best_state)
    return model, best_val_auc


def eval_loader_single(loader, model, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb   = xb.to(device)
            prob = torch.sigmoid(model(xb)).cpu().numpy()
            ps.append(prob)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)



def ensemble_predict(models, loader, device):
"""
Collect the predicted probabilities from all models (shape: (n_models, n_samples)).
Apply IQR filtering to each sample, then compute the mean.
"""
    all_probs = []
    for model in models:
        model.eval()
        ps = []
        with torch.no_grad():
            for xb, _ in loader:
                xb   = xb.to(device)
                prob = torch.sigmoid(model(xb)).cpu().numpy()
                ps.append(prob)
        all_probs.append(np.concatenate(ps))

    all_probs = np.stack(all_probs, axis=0)  # (n_models, n_samples)

    q1  = np.percentile(all_probs, 25, axis=0)
    q3  = np.percentile(all_probs, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    final = np.zeros(all_probs.shape[1])
    for i in range(all_probs.shape[1]):
        col  = all_probs[:, i]
        mask = (col >= lower[i]) & (col <= upper[i])
        final[i] = col[mask].mean() if mask.any() else col.mean()

    return final


def eval_ensemble(models, loader, device):
    y_prob = ensemble_predict(models, loader, device)
    y_true = np.concatenate([yb.numpy() for _, yb in loader])
    auc    = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    return auc, auprc


def run_ptm(ptm_name, train_df, test_df, device,
            val_ratio=0.1, num_models=NUM_MODELS):
"""
Train 10 models for a single PTM type and evaluate them via ensembling.
val_ratio: The proportion of the training data allocated for validation (9% in the original version).
"""
    print(f"\n{'='*60}")
    print(f"PTM: {ptm_name}")
    print(f"{'='*60}")

    n_val   = int(len(train_df) * val_ratio)
    idx     = np.random.permutation(len(train_df))
    val_df  = train_df.iloc[idx[:n_val]].reset_index(drop=True)
    trn_df  = train_df.iloc[idx[n_val:]].reset_index(drop=True)

    trn_ds  = PTMDataset(trn_df)
    val_ds  = PTMDataset(val_df)
    tst_ds  = PTMDataset(test_df)

    trn_loader = DataLoader(trn_ds, batch_size=BATCH_TRAIN,
                            shuffle=True,  num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_TEST,
                            shuffle=False, num_workers=2, pin_memory=True)
    tst_loader = DataLoader(tst_ds, batch_size=BATCH_TEST,
                            shuffle=False, num_workers=2, pin_memory=True)

    models     = []
    val_aucs   = []
    for seed in range(num_models):
        print(f"\n  Training model {seed+1}/{num_models} (seed={seed})...")
        model, val_auc = train_one_model(
            trn_loader, val_loader, seed=seed, device=device
        )
        models.append(model)
        val_aucs.append(val_auc)
        print(f"  Model {seed+1} val AUC: {val_auc:.4f}")

    ens_auc, ens_auprc = eval_ensemble(models, tst_loader, device)
    print(f"\n[{ptm_name}] Ensemble Test AUROC={ens_auc:.4f}, AUPRC={ens_auprc:.4f}")
    print(f"  Individual val AUCs: {[f'{v:.4f}' for v in val_aucs]}")

    return models, ens_auc, ens_auprc


if __name__ == "__main__":
    BASE = "/home/FCAM/juli/HRP/retrain"

    PTM_TASKS = [
        ("phosphorylation_st", "phosphorylation_st"),
        ("phosphorylation_y",  "phosphorylation_y"),
        ("sumoylation_k",      "sumoylation_k"),
        ("ubiquitination_k",   "ubiquitination_k"),
        ("acetylation_k",      "acetylation_k"),
        ("glycosylation_n",    "glycosylation_n"),
        ("methylation_k",      "methylation_k"),
        ("methylation_r",      "methylation_r"),
    ]

    results = {}
    for ptm_name, file_stem in PTM_TASKS:
        train_df = pd.read_csv(f"{BASE}/{file_stem}_train.tsv", sep="\t")
        test_df  = pd.read_csv(f"{BASE}/{file_stem}_test.tsv",  sep="\t")

        models, auc, auprc = run_ptm(
            ptm_name, train_df, test_df, device=DEVICE
        )
        results[ptm_name] = {"AUROC": auc, "AUPRC": auprc}

        os.makedirs(f"checkpoints/{ptm_name}", exist_ok=True)
        for i, m in enumerate(models):
            torch.save(m.state_dict(),
                       f"checkpoints/{ptm_name}/model_{i}.pt")

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for ptm, res in results.items():
        print(f"{ptm:25s}  AUROC={res['AUROC']:.4f}  AUPRC={res['AUPRC']:.4f}")
