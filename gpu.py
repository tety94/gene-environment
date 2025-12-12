# gpu_pipeline.py
import pandas as pd
import numpy as np
import random
import pickle
import os
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from cuml.preprocessing import StandardScaler as cuStandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from statsmodels.stats.multitest import multipletests
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from db import gene_already_done, save_gene_result, get_conn

# ---------------- CONFIG ----------------
raw_file = "gen_diminuito.csv"
env_file = "componenti_ambientali.csv"
sep = ';'
decimal = '.'
onset_col = "onset_age"
exposures = ["seminativi_1500"]
covariates = ["sex", "onset_site", "diagnostic_delay"]
n_perm = 1000
random_state = 42
standardize = True
min_treated = 5
min_sample_size = 10
max_workers = 2  # parallelismo CPU
match_k = 3
TEMP_DF_PATH = "temp_df.pkl"
# ----------------------------------------

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Usando device: {device}, GPU disponibili: {num_gpus}")

# ---------------- HELPERS ----------------
def _prepare_matching_matrix(df, cols):
    """Prepara la matrice per cuML NearestNeighbors"""
    if not cols:
        raise ValueError("No columns for matching")
    features = []
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(df[c].fillna(df[c].mean()))
        else:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
            features.append(dummies)
    X = pd.concat(features, axis=1)
    X_scaled = cuStandardScaler().fit_transform(X.values)
    return X_scaled, X.columns

def match_control_units(df, gene_col, k=2, covariates_for_matching=None):
    if covariates_for_matching is None:
        raise ValueError("Serve almeno una covariata per il matching")
    treated = df[df[gene_col] == 1].reset_index(drop=True)
    control = df[df[gene_col] == 0].reset_index(drop=True)
    if treated.shape[0] == 0 or control.shape[0] == 0:
        return None

    df_matching = pd.concat([treated, control], ignore_index=True)
    X_scaled, _ = _prepare_matching_matrix(df_matching, covariates_for_matching)

    # split
    mask_t = df_matching[gene_col] == 1
    X_t = X_scaled[mask_t.values]
    X_c = X_scaled[~mask_t.values]

    k_used = min(k, X_c.shape[0])
    nn = cuNearestNeighbors(n_neighbors=k_used)
    nn.fit(X_c)
    distances, indices = nn.kneighbors(X_t)

    selected_ctrl_pos = np.unique(indices.flatten())
    ctrl_idx = df_matching[~mask_t].index[selected_ctrl_pos]
    matched_controls = df_matching.loc[ctrl_idx]
    matched_treated = df_matching.loc[mask_t]
    matched_df = pd.concat([matched_treated, matched_controls], ignore_index=True)
    return matched_df

# ---------------- PyTorch MODEL ----------------
class InteractionModel(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)
    def forward(self, x):
        return self.linear(x)

def fit_interaction_model(X, y):
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y.values, dtype=torch.float32).view(-1,1).to(device)
    model = InteractionModel(X.shape[1]).to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for _ in range(500):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    # ritorna i coefficienti (w e bias)
    coefs = model.module.linear.weight.detach().cpu().numpy().flatten() if num_gpus > 1 else model.linear.weight.detach().cpu().numpy().flatten()
    bias = model.module.linear.bias.item() if num_gpus > 1 else model.linear.bias.item()
    return coefs, bias

# ---------------- MAIN ----------------
def process_single_gene(gene_col, gene_original, Ecols):
    print(f"[START] {gene_original}")
    df = pickle.load(open(TEMP_DF_PATH, "rb"))
    conn = get_conn()
    if gene_already_done(conn, gene_original):
        conn.close()
        return

    n_treated = int((df[gene_col]==1).sum())
    n_control = int((df[gene_col]==0).sum())
    if n_treated < min_treated or n_control == 0:
        conn.close()
        return

    cols = [onset_col, gene_col] + Ecols + covariates
    df_model = df[cols].dropna()
    if df_model.shape[0] < min_sample_size:
        conn.close()
        return

    cov_match = Ecols + covariates
    matched_obs = match_control_units(df_model, gene_col, k=match_k, covariates_for_matching=cov_match)
    if matched_obs is None or matched_obs.shape[0] < min_sample_size:
        conn.close()
        return

    X_cols = Ecols + [gene_col] + [f"{gene_col}:{e}" for e in Ecols] + covariates
    X = matched_obs[Ecols + [gene_col]].values  # semplificazione
    y = matched_obs[onset_col]

    try:
        coefs, bias = fit_interaction_model(X, y)
        obs_coef = coefs[0]  # coefficiente gene*exposure
    except Exception as e:
        obs_coef = None

    rng = np.random.RandomState(random_state + (abs(hash(gene_col)) % 2_000_000))
    perm_betas = []
    for _ in range(n_perm):
        df_perm = df_model.copy()
        df_perm[gene_col] = rng.permutation(df_perm[gene_col].values)
        matched_perm = match_control_units(df_perm, gene_col, k=match_k, covariates_for_matching=cov_match)
        if matched_perm is None or matched_perm.shape[0] < min_sample_size:
            perm_betas.append(np.nan)
            continue
        Xp = matched_perm[Ecols + [gene_col]].values
        yp = matched_perm[onset_col]
        try:
            coefs_perm, _ = fit_interaction_model(Xp, yp)
            perm_betas.append(coefs_perm[0])
        except:
            perm_betas.append(np.nan)

    perm_betas = np.array([x for x in perm_betas if not np.isnan(x)])
    p_emp = float(np.mean(np.abs(perm_betas) >= np.abs(obs_coef))) if perm_betas.size>0 else None

    save_gene_result(conn, gene_original,
                     int(matched_obs[gene_col].sum()),
                     int((matched_obs[gene_col]==0).sum()),
                     obs_coef,
                     float(np.mean(perm_betas)) if perm_betas.size>0 else None,
                     float(np.std(perm_betas)) if perm_betas.size>0 else None,
                     p_emp)
    conn.close()
    print(f"[DONE] {gene_original}")

# ---------------- SCRIPT PRINCIPALE ----------------
def main():
    df_gen = pd.read_csv(raw_file, sep=sep, decimal=decimal)
    non_gen_cols = ["FID","IID","PAT","MAT","SEX","PHENOTYPE","id"]
    df_gen = df_gen.loc[:, (df_gen==-1).mean() < 0.30]
    gene_cols = [c for c in df_gen.columns if c not in non_gen_cols]
    for g in gene_cols:
        df_gen[g] = (df_gen[g]>0).astype(int)
    if "IID" in df_gen.columns:
        df_gen = df_gen.rename(columns={"IID":"id"})
    df_env = pd.read_csv(env_file, sep=sep, decimal=decimal)
    df_env["sex"] = df_env["sex"].astype("category")
    df_env["onset_site"] = df_env["onset_site"].astype("category")
    df = pd.merge(df_env, df_gen, on="id", how="inner")
    df[onset_col] = pd.to_numeric(df[onset_col], errors="coerce")

    Ecols = []
    for exp in exposures:
        df[exp] = pd.to_numeric(df[exp], errors="coerce")
        if standardize:
            df[exp+"_std"] = cuStandardScaler().fit_transform(df[[exp]].values)
            Ecols.append(exp+"_std")
        else:
            Ecols.append(exp)

    safe = {g:f"gene_{i}" for i,g in enumerate(gene_cols)}
    df.rename(columns=safe, inplace=True)
    mapping = {v:k for k,v in safe.items()}
    df.to_pickle(TEMP_DF_PATH)

    # Parallel CPU per process_single_gene
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_single_gene, gc, mapping[gc], Ecols) for gc in list(safe.values())]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("Errore processo:", e)

if __name__=="__main__":
    main()
