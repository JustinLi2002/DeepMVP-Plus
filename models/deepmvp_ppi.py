import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import copy
import os


DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_MODELS  = 10
MAX_EPOCHS  = 100
PATIENCE    = 10
BATCH_TRAIN = 64
BATCH_TEST  = 512
LR          = 1e-3

AA_LIST   = list("ACDEFGHIKLMNPQRSTVWY") + ["U", "O", "X"]
aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}
C         = len(AA_LIST)  # 23
EYE       = np.eye(C, dtype=np.float32)

print(f"Using device: {DEVICE}")

#  Dataset（含PPI）
class PTMDataset_PPI(Dataset):
    def __init__(self, df):
        self.seq = df["x"].values
        self.ppi = df["PPI"].values
        self.y   = df["y"].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        s     = self.seq[i]
        idx   = [aa_to_idx[a] for a in s]
        x_seq = EYE[idx].T.astype(np.float32)             
        x_ppi = np.asarray(self.ppi[i], dtype=np.float32) 
        return (torch.from_numpy(x_seq),
                torch.from_numpy(x_ppi),
                torch.tensor(self.y[i]))


class DeepMVP_PPI(nn.Module):
"""
CNN+BiGRU backbone identical to the baseline
Additional PPI branch: FC(605→128) → LeakyReLU
Fusion: cat([gru_out(100), ppi_out(128)]) → FC(228→64) → out
"""
    def __init__(self, in_ch=23, ppi_dim=605, dropout=0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, 512, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 128, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(128)
        self.drop  = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=50,
            batch_first=True,
            bidirectional=True
        )  # → (batch, 100)

        self.ppi_fc = nn.Linear(ppi_dim, 128)

        self.fc1    = nn.Linear(228, 64)
        self.bn4    = nn.BatchNorm1d(64)
        self.drop2  = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x_seq, x_ppi):
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(x_seq))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.drop(F.leaky_relu(self.bn3(self.conv3(x))))
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        x = torch.cat([h[0], h[1]], dim=1)   # (batch, 100)

        p = F.leaky_relu(self.ppi_fc(x_ppi))  # (batch, 128)

        x = torch.cat([x, p], dim=1)           # (batch, 228)
        x = self.drop2(F.leaky_relu(self.bn4(self.fc1(x))))
        return self.fc_out(x).squeeze(-1)


def eval_loader(loader, model, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb_seq, xb_ppi, yb in loader:
            xb_seq = xb_seq.to(device)
            xb_ppi = xb_ppi.to(device)
            prob   = torch.sigmoid(model(xb_seq, xb_ppi)).cpu().numpy()
            ps.append(prob)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)


def train_one_model(trn_loader, val_loader, seed, ppi_dim, device,
                    max_epochs=MAX_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = DeepMVP_PPI(ppi_dim=ppi_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    ys         = trn_loader.dataset.y
    neg, pos   = (ys == 0).sum(), (ys == 1).sum()
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_state   = None
    wait         = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb_seq, xb_ppi, yb in trn_loader:
            xb_seq, xb_ppi, yb = xb_seq.to(device), xb_ppi.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb_seq, xb_ppi), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_auc, _ = eval_loader(val_loader, model, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = copy.deepcopy(model.state_dict())
            wait         = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop @ epoch {epoch}, best val AUC={best_val_auc:.4f}")
                break

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d} | loss={total_loss/len(trn_loader):.4f} | val AUC={val_auc:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_auc


# ensemble
def ensemble_predict(models, loader, device):
    all_probs = []
    for model in models:
        model.eval()
        ps = []
        with torch.no_grad():
            for xb_seq, xb_ppi, _ in loader:
                xb_seq = xb_seq.to(device)
                xb_ppi = xb_ppi.to(device)
                prob   = torch.sigmoid(model(xb_seq, xb_ppi)).cpu().numpy()
                ps.append(prob)
        all_probs.append(np.concatenate(ps))

    all_probs = np.stack(all_probs, axis=0)  # (10, n_samples)

    q1    = np.percentile(all_probs, 25, axis=0)
    q3    = np.percentile(all_probs, 75, axis=0)
    iqr   = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    final = np.zeros(all_probs.shape[1])
    for i in range(all_probs.shape[1]):
        col      = all_probs[:, i]
        mask     = (col >= lower[i]) & (col <= upper[i])
        final[i] = col[mask].mean() if mask.any() else col.mean()

    return final


def eval_ensemble(models, loader, device):
    y_prob = ensemble_predict(models, loader, device)
    y_true = np.concatenate([yb.numpy() for _, _, yb in loader])
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)


#slowest part
def load_ppi(feat_path, ids_path):
    protein_features = np.load(feat_path)
    with open(ids_path, "r") as f:
        protein_ids = json.load(f)
    protein_to_vec = dict(zip(protein_ids, protein_features))
    ppi_dim        = protein_features.shape[1]
    zero_vec       = np.zeros(ppi_dim, dtype=np.float32)
    return protein_to_vec, ppi_dim, zero_vec


def add_ppi(df, protein_to_vec, zero_vec):
    df = df.copy()
    df["PPI"] = df["protein"].map(protein_to_vec)
    df["PPI"] = df["PPI"].apply(
        lambda v: np.asarray(v, dtype=np.float32)
        if isinstance(v, np.ndarray) else zero_vec
    )
    return df


def run_ptm(ptm_name, train_df, test_df, ppi_dim, device,
            val_ratio=0.1, num_models=NUM_MODELS):

    print(f"\n{'='*60}")
    print(f"PTM: {ptm_name}")
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")
    print(f"{'='*60}")

    np.random.seed(42)
    idx    = np.random.permutation(len(train_df))
    n_val  = int(len(train_df) * val_ratio)
    val_df = train_df.iloc[idx[:n_val]].reset_index(drop=True)
    trn_df = train_df.iloc[idx[n_val:]].reset_index(drop=True)

    trn_ds  = PTMDataset_PPI(trn_df)
    val_ds  = PTMDataset_PPI(val_df)
    tst_ds  = PTMDataset_PPI(test_df)

    pin = DEVICE.type == "cuda"
    trn_loader = DataLoader(trn_ds, batch_size=BATCH_TRAIN,
                            shuffle=True,  num_workers=4, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_TEST,
                            shuffle=False, num_workers=4, pin_memory=pin)
    tst_loader = DataLoader(tst_ds, batch_size=BATCH_TEST,
                            shuffle=False, num_workers=4, pin_memory=pin)

    models   = []
    val_aucs = []
    for seed in range(num_models):
        print(f"\n  [Model {seed+1}/{num_models}] seed={seed}")
        model, val_auc = train_one_model(
            trn_loader, val_loader, seed=seed,
            ppi_dim=ppi_dim, device=device
        )
        models.append(model)
        val_aucs.append(val_auc)
        print(f"  → Best val AUC: {val_auc:.4f}")

    ens_auc, ens_auprc = eval_ensemble(models, tst_loader, device)
    print(f"\n[{ptm_name}] Ensemble  AUROC={ens_auc:.4f}  AUPRC={ens_auprc:.4f}")
    print(f"  Individual val AUCs: {[f'{v:.4f}' for v in val_aucs]}")

    ckpt_dir = f"checkpoints_ppi/{ptm_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    for i, m in enumerate(models):
        torch.save(m.state_dict(), f"{ckpt_dir}/model_{i}.pt")
    print(f"  Checkpoints saved to {ckpt_dir}/")

    return ens_auc, ens_auprc


if __name__ == "__main__":

    BASE      = "/home/FCAM/juli/HRP/retrain"
    FEAT_PATH = "/home/FCAM/juli/HRP/notebooks/protein_features_ppi.npy"
    IDS_PATH  = "/home/FCAM/juli/HRP/notebooks/protein_ids_ppi.json"

    # load PPI data
    protein_to_vec, ppi_dim, zero_vec = load_ppi(FEAT_PATH, IDS_PATH)
    print(f"PPI dim: {ppi_dim}, proteins: {len(protein_to_vec)}")

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
        train_path = f"{BASE}/{file_stem}_train.tsv"
        test_path  = f"{BASE}/{file_stem}_test.tsv"

        if not os.path.exists(train_path):
            print(f"[SKIP] {train_path} not found")
            continue

        train_df = pd.read_csv(train_path, sep="\t")
        test_df  = pd.read_csv(test_path,  sep="\t")

        # add PPI feature
        train_df = add_ppi(train_df, protein_to_vec, zero_vec)
        test_df  = add_ppi(test_df,  protein_to_vec, zero_vec)

        auc, auprc = run_ptm(ptm_name, train_df, test_df,
                             ppi_dim=ppi_dim, device=DEVICE)
        results[ptm_name] = {"AUROC": auc, "AUPRC": auprc}

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY (+PPI)")
    print("="*60)
    print(f"{'PTM':<25}  {'AUROC':>7}  {'AUPRC':>7}")
    print("-"*45)
    for ptm, res in results.items():
        print(f"{ptm:<25}  {res['AUROC']:>7.4f}  {res['AUPRC']:>7.4f}")
