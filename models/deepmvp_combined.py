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

# ---------------------------------------------------------
# 1. Dataset (seq + node2vec PPI + kinase scores)
# ---------------------------------------------------------
class PTMDataset_Combined(Dataset):
    def __init__(self, df):
        self.seq    = df["x"].values
        self.ppi    = df["PPI"].values      # node2vec 128-dim
        self.kinase = df["Kinase"].values   # kinase scores 605-dim
        self.y      = df["y"].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        s      = self.seq[i]
        idx    = [aa_to_idx[a] for a in s]
        x_seq  = EYE[idx].T.astype(np.float32)
        x_ppi  = np.asarray(self.ppi[i],    dtype=np.float32)
        x_kin  = np.asarray(self.kinase[i], dtype=np.float32)
        return (torch.from_numpy(x_seq),
                torch.from_numpy(x_ppi),
                torch.from_numpy(x_kin),
                torch.tensor(self.y[i]))


# ---------------------------------------------------------
# 2. Model: CNN+BiGRU + node2vec branch + kinase branch
# ---------------------------------------------------------
class DeepMVP_Combined(nn.Module):
    """
    Sequence branch:   CNN+BiGRU → (100,)
    node2vec branch:   FC(128→64) → LeakyReLU → (64,)
    Kinase branch:     FC(605→64) → LeakyReLU → (64,)
    Fusion:            cat([100, 64, 64]) → (228,) → FC(228→64) → out
    """
    def __init__(self, in_ch=23, ppi_dim=128, kinase_dim=605, dropout=0.3):
        super().__init__()

        # Sequence backbone (identical to baseline)
        self.conv1 = nn.Conv1d(in_ch, 512, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 128, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(128)
        self.drop  = nn.Dropout(dropout)
        self.gru   = nn.GRU(128, 50, batch_first=True, bidirectional=True)
        # → (batch, 100)

        # node2vec PPI branch
        self.ppi_fc = nn.Linear(ppi_dim, 64)

        # Kinase scores branch
        self.kin_fc = nn.Linear(kinase_dim, 64)

        # Fusion: 100 + 64 + 64 = 228
        self.fc1    = nn.Linear(228, 64)
        self.bn4    = nn.BatchNorm1d(64)
        self.drop2  = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x_seq, x_ppi, x_kin):
        # Sequence branch
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(x_seq))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.drop(F.leaky_relu(self.bn3(self.conv3(x))))
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        x = torch.cat([h[0], h[1]], dim=1)       # (batch, 100)

        # PPI branch
        p = F.leaky_relu(self.ppi_fc(x_ppi))     # (batch, 64)

        # Kinase branch
        k = F.leaky_relu(self.kin_fc(x_kin))     # (batch, 64)

        # Fusion
        x = torch.cat([x, p, k], dim=1)           # (batch, 228)
        x = self.drop2(F.leaky_relu(self.bn4(self.fc1(x))))
        return self.fc_out(x).squeeze(-1)


# ---------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------
def eval_loader(loader, model, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb_seq, xb_ppi, xb_kin, yb in loader:
            xb_seq = xb_seq.to(device)
            xb_ppi = xb_ppi.to(device)
            xb_kin = xb_kin.to(device)
            prob   = torch.sigmoid(model(xb_seq, xb_ppi, xb_kin)).cpu().numpy()
            ps.append(prob)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)


# ---------------------------------------------------------
# 4. Train single model
# ---------------------------------------------------------
def train_one_model(trn_loader, val_loader, seed, device,
                    max_epochs=MAX_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = DeepMVP_Combined().to(device)
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
        for xb_seq, xb_ppi, xb_kin, yb in trn_loader:
            xb_seq = xb_seq.to(device)
            xb_ppi = xb_ppi.to(device)
            xb_kin = xb_kin.to(device)
            yb     = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb_seq, xb_ppi, xb_kin), yb)
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


# ---------------------------------------------------------
# 5. Ensemble
# ---------------------------------------------------------
def ensemble_predict(models, loader, device):
    all_probs = []
    for model in models:
        model.eval()
        ps = []
        with torch.no_grad():
            for xb_seq, xb_ppi, xb_kin, _ in loader:
                xb_seq = xb_seq.to(device)
                xb_ppi = xb_ppi.to(device)
                xb_kin = xb_kin.to(device)
                prob   = torch.sigmoid(model(xb_seq, xb_ppi, xb_kin)).cpu().numpy()
                ps.append(prob)
        all_probs.append(np.concatenate(ps))

    all_probs = np.stack(all_probs, axis=0)

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
    y_true = np.concatenate([yb.numpy() for _, _, _, yb in loader])
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)


# ---------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------
def load_features(feat_path, ids_path):
    vecs     = np.load(feat_path)
    with open(ids_path) as f:
        ids  = json.load(f)
    feat_map = dict(zip(ids, vecs))
    zero_vec = np.zeros(vecs.shape[1], dtype=np.float32)
    return feat_map, zero_vec


def add_feature(df, col_name, feat_map, zero_vec):
    df = df.copy()
    df[col_name] = df["protein"].map(feat_map)
    df[col_name] = df[col_name].apply(
        lambda v: np.asarray(v, dtype=np.float32)
        if isinstance(v, np.ndarray) else zero_vec
    )
    return df


def run_ptm(ptm_name, train_df, test_df, device,
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

    trn_ds = PTMDataset_Combined(trn_df)
    val_ds = PTMDataset_Combined(val_df)
    tst_ds = PTMDataset_Combined(test_df)

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
        model, val_auc = train_one_model(trn_loader, val_loader,
                                         seed=seed, device=device)
        models.append(model)
        val_aucs.append(val_auc)
        print(f"  Best val AUC: {val_auc:.4f}")

    ens_auc, ens_auprc = eval_ensemble(models, tst_loader, device)
    print(f"\n[{ptm_name}] Ensemble  AUROC={ens_auc:.4f}  AUPRC={ens_auprc:.4f}")
    print(f"  Individual val AUCs: {[f'{v:.4f}' for v in val_aucs]}")

    ckpt_dir = f"checkpoints_combined/{ptm_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    for i, m in enumerate(models):
        torch.save(m.state_dict(), f"{ckpt_dir}/model_{i}.pt")
    print(f"  Checkpoints saved to {ckpt_dir}/")

    return ens_auc, ens_auprc


# ---------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------
if __name__ == "__main__":

    BASE      = "/home/FCAM/juli/HRP/retrain"

    # node2vec PPI embeddings (128-dim)
    PPI_FEAT  = "/home/FCAM/juli/HRP/notebooks/protein_features_ppi.npy"
    PPI_IDS   = "/home/FCAM/juli/HRP/notebooks/protein_ids_ppi.json"

    # Kinase prediction scores (605-dim)
    KIN_FEAT  = "/home/FCAM/juli/HRP/notebooks/protein_features.npy"
    KIN_IDS   = "/home/FCAM/juli/HRP/notebooks/protein_ids.json"

    print("Loading node2vec PPI embeddings...")
    ppi_map, ppi_zero = load_features(PPI_FEAT, PPI_IDS)
    print(f"  {len(ppi_map)} proteins, dim={list(ppi_map.values())[0].shape[0]}")

    print("Loading kinase scores...")
    kin_map, kin_zero = load_features(KIN_FEAT, KIN_IDS)
    print(f"  {len(kin_map)} proteins, dim={list(kin_map.values())[0].shape[0]}")

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

        # Add both feature types
        train_df = add_feature(train_df, "PPI",    ppi_map, ppi_zero)
        test_df  = add_feature(test_df,  "PPI",    ppi_map, ppi_zero)
        train_df = add_feature(train_df, "Kinase", kin_map, kin_zero)
        test_df  = add_feature(test_df,  "Kinase", kin_map, kin_zero)

        auc, auprc = run_ptm(ptm_name, train_df, test_df, device=DEVICE)
        results[ptm_name] = {"AUROC": auc, "AUPRC": auprc}

    print("\n" + "="*60)
    print("FINAL SUMMARY (+PPI +Kinase Combined)")
    print("="*60)
    print(f"{'PTM':<25}  {'AUROC':>7}  {'AUPRC':>7}")
    print("-"*45)
    for ptm, res in results.items():
        print(f"{ptm:<25}  {res['AUROC']:>7.4f}  {res['AUPRC']:>7.4f}")
