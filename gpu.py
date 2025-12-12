import pandas as pd
import numpy as np
import torch
import random
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import os
from db import gene_already_done, save_gene_result, get_conn
from sklearn.preprocessing import StandardScaler

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
match_k = 3
TEMP_DF_PATH = "temp_df.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)

# ---------------- HELPERS ----------------

def standardize_tensor(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True)
    return (x - mean) / (std + 1e-8)

def one_hot_tensor(df, col):
    cats = df[col].astype("category").cat.categories
    oh = torch.zeros((len(df), len(cats)-1), device=device)
    cat_map = {cat:i for i, cat in enumerate(cats)}
    for i, val in enumerate(df[col]):
        idx = cat_map[val]
        if idx > 0:  # drop first
            oh[i, idx-1] = 1
    return oh

def prepare_matrix(df, cols):
    mats = []
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            tensor = torch.tensor(df[c].fillna(df[c].mean()).values, dtype=torch.float32, device=device).unsqueeze(1)
            mats.append(tensor)
        else:
            mats.append(one_hot_tensor(df, c))
    return torch.cat(mats, dim=1) if mats else None

def match_control_units(df, gene_col, k=3, covariates_for_matching=None):
    treated = df[df[gene_col]==1].reset_index(drop=True)
    control = df[df[gene_col]==0].reset_index(drop=True)
    if treated.shape[0]==0 or control.shape[0]==0:
        return None
    df_all = pd.concat([treated, control], ignore_index=True)
    X = prepare_matrix(df_all, covariates_for_matching)
    mask_t = torch.tensor(df_all[gene_col].values==1, device=device)
    X_t = X[mask_t]
    X_c = X[~mask_t]
    # distanza euclidea
    dists = torch.cdist(X_t, X_c)
    topk = min(k, X_c.shape[0])
    indices = torch.topk(-dists, k=topk, dim=1).indices
    ctrl_idx = torch.unique(indices.flatten())
    matched_controls = df_all.iloc[ctrl_idx.cpu().numpy()]
    matched_treated = df_all[mask_t.cpu().numpy()]
    return pd.concat([matched_treated, matched_controls], ignore_index=True)

def build_interaction_matrix(df, gene_col, Ecols, covariates):
    tensors = []
    gene_tensor = torch.tensor(df[gene_col].values, device=device, dtype=torch.float32).unsqueeze(1)
    tensors.append(gene_tensor)
    for e in Ecols:
        t = torch.tensor(df[e].values, device=device, dtype=torch.float32).unsqueeze(1)
        tensors.append(t)
        tensors.append(t * gene_tensor)  # interazione
    for c in covariates:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            t = torch.tensor(df[c].fillna(df[c].mean()).values, device=device, dtype=torch.float32).unsqueeze(1)
            tensors.append(t)
        else:
            tensors.append(one_hot_tensor(df, c))
    return torch.cat(tensors, dim=1)

def linear_regression(X, y):
    # X [N, p], y [N]
    X_ = torch.cat([torch.ones(X.shape[0],1, device=device), X], dim=1)
    beta = torch.linalg.lstsq(X_, y.unsqueeze(1)).solution.squeeze(1)
    return beta

def process_single_gene(gene_col, gene_original, Ecols):
    df = pickle.load(open(TEMP_DF_PATH, "rb"))
    conn = get_conn()
    if gene_already_done(conn, gene_original):
        conn.close()
        return None
    n_treated = int((df[gene_col]==1).sum())
    n_control = int((df[gene_col]==0).sum())
    if n_treated < min_treated or n_control==0:
        conn.close()
        return None
    cols = [onset_col, gene_col]+Ecols+covariates
    df_model = df[cols].dropna()
    if df_model.shape[0]<min_sample_size:
        conn.close()
        return None
    cov_match = Ecols + covariates
    matched_obs = match_control_units(df_model, gene_col, k=match_k, covariates_for_matching=cov_match)
    if matched_obs is None or matched_obs.shape[0]<min_sample_size:
        conn.close()
        return None
    X_obs = build_interaction_matrix(matched_obs, gene_col, Ecols, covariates)
    y_obs = torch.tensor(matched_obs[onset_col].values, device=device, dtype=torch.float32)
    beta_obs = linear_regression(X_obs, y_obs)
    # il coefficiente dell'interazione è l'ultima colonna dei termini interazione
    obs_coef = float(beta_obs[len(Ecols)+1])
    # permutation test
    perm_betas = []
    gene_tensor_orig = torch.tensor(df_model[gene_col].values, device=device, dtype=torch.float32)
    y_tensor_orig = torch.tensor(df_model[onset_col].values, device=device, dtype=torch.float32)
    for _ in range(n_perm):
        perm_gene = gene_tensor_orig[torch.randperm(len(gene_tensor_orig))]
        df_model[gene_col] = perm_gene.cpu().numpy()
        matched_perm = match_control_units(df_model, gene_col, k=match_k, covariates_for_matching=cov_match)
        if matched_perm is None or matched_perm.shape[0]<min_sample_size:
            perm_betas.append(np.nan)
            continue
        X_perm = build_interaction_matrix(matched_perm, gene_col, Ecols, covariates)
        y_perm = torch.tensor(matched_perm[onset_col].values, device=device, dtype=torch.float32)
        beta_perm = linear_regression(X_perm, y_perm)
        perm_betas.append(float(beta_perm[len(Ecols)+1]))
    perm_betas = np.array([x for x in perm_betas if not np.isnan(x)])
    p_emp = float(np.mean(np.abs(perm_betas)>=np.abs(obs_coef))) if perm_betas.size>0 else None
    save_gene_result(conn, gene_original, int(matched_obs[gene_col].sum()),
                     int((matched_obs[gene_col]==0).sum()), obs_coef,
                     float(np.mean(perm_betas)) if perm_betas.size>0 else None,
                     float(np.std(perm_betas)) if perm_betas.size>0 else None,
                     p_emp)
    conn.close()
    return gene_original

# ---------- MAIN ----------

def main():
    df_gen = pd.read_csv(raw_file, sep=sep, decimal=decimal)
    non_gen_cols = ["FID","IID","PAT","MAT","SEX","PHENOTYPE","id"]
    df_gen = df_gen.loc[:, (df_gen==-1).mean()<0.30]
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
            df[exp+"_std"] = StandardScaler().fit_transform(df[[exp]])
            Ecols.append(exp+"_std")
        else:
            Ecols.append(exp)
    safe = {g:f"gene_{i}" for i,g in enumerate(gene_cols)}
    df.rename(columns=safe, inplace=True)
    gene_cols_safe = list(safe.values())
    mapping = {v:k for k,v in safe.items()}
    df.to_pickle(TEMP_DF_PATH)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(process_single_gene, gc, mapping[gc], Ecols) for gc in gene_cols_safe]
        for f in as_completed(futures):
            try: f.result()
            except Exception as e: print("Errore:", e)
    print("Done")

if __name__=="__main__":
    main()
