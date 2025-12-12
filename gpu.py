#!/usr/bin/env python3
# gpu_full_pipeline.py
"""
Pipeline Gene x Environment totalmente GPU-accelerata (PyTorch + RAPIDS/cuML).

Prerequisiti consigliati (CUDA 11.5):
- torch (con supporto CUDA 11.5)
- cudf, cuml (RAPIDS build per CUDA 11.5)
- pandas, numpy, matplotlib, tqdm
- il tuo modulo db.py con: gene_already_done, save_gene_result, load_gene_results, get_conn

Se mancanti, lo script proverà a funzionare in CPU-fallback mode.
"""

import os
import sys
import math
import pickle
import random
import warnings
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from statsmodels.stats.multitest import multipletests

# --- GPU libraries (try imports, fallback to CPU) ---
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# RAPIDS (cudf + cuml) for GPU dataframes + nearest neighbors
try:
    import cudf
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    RAPIDS_AVAILABLE = True
except Exception:
    RAPIDS_AVAILABLE = False

# Scikit-learn fallback for matching if cuML not available
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors as skNearestNeighbors
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# db functions provided by user
try:
    from db import gene_already_done, load_gene_results, save_gene_result, get_conn
except Exception as e:
    # If db.py is missing we create no-op placeholders to allow running without DB for testing.
    warnings.warn(f"db.py import failed: {e}. Using local placeholder functions.")
    def gene_already_done(conn, gene): return False
    def load_gene_results(): return pd.DataFrame()
    def save_gene_result(conn, gene, n_treated, n_controls, obs_coef, perm_mean, perm_std, p_emp):
        print(f"[SAVE] {gene} treated={n_treated} controls={n_controls} coef={obs_coef} p={p_emp}")
    def get_conn(): return None

# ---------------- CONFIG ----------------
raw_file = "gen_diminuito.csv"
env_file = "componenti_ambientali.csv"
sep = ';'
decimal = '.'
onset_col = "onset_age"
exposures = ["seminativi_1500"]
covariates = ["sex", "onset_site", "diagnostic_delay"]

# Permutation and resource config
n_perm = 1000
random_state = 42
standardize = True
min_treated = 5
min_sample_size = 10
TEMP_DF_PATH = "temp_df.pkl"
match_k = 3

# Torch device
if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = None

# seeds
np.random.seed(random_state)
random.seed(random_state)
if TORCH_AVAILABLE:
    torch.manual_seed(random_state)

# ----------------- UTILITIES -----------------

def info(msg):
    print(msg)

def warn(msg):
    print("[WARN]", msg)

def ensure_cols(df: pd.DataFrame, cols: List[str]):
    return [c for c in cols if c in df.columns]

# ----------------- DATAFRAME / GPU helpers -----------------
def to_cudf_if_possible(df: pd.DataFrame):
    """Convert pandas DataFrame to cudf.DataFrame if RAPIDS available, otherwise return pandas."""
    if RAPIDS_AVAILABLE:
        return cudf.DataFrame.from_pandas(df)
    return df

def to_pandas_if_needed(df):
    if RAPIDS_AVAILABLE and isinstance(df, cudf.DataFrame):
        return df.to_pandas()
    return df

# ----------------- MATCHING (GPU with cuML, fallback to sklearn CPU) -----------------

def prepare_matching_matrix_pandas(df: pd.DataFrame, cols: List[str]):
    """
    Build numeric matrix (pandas) for matching:
    - numeric columns as-is (NaN -> mean)
    - categorical columns one-hot encoded (drop_first=True)
    Returns numpy array (float32).
    """
    features = []
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            arr = df[c].fillna(df[c].mean()).astype(float)
            features.append(arr)
        else:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
            features.append(dummies)
    if not features:
        raise ValueError("No valid matching features")
    X = pd.concat(features, axis=1)
    X = X.astype(float)
    # scale for numerical stability
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    return X.values.astype(np.float32), X.columns.tolist()

def prepare_matching_matrix_cudf(df, cols: List[str]):
    """
    Build numeric matrix (cudf) for matching:
    - numeric columns: fillna with mean (cudf)
    - categoricals: get_dummies (cudf)
    Returns cudf.DataFrame
    """
    parts = []
    for c in cols:
        if c not in df.columns:
            continue
        # cudf supports is_numeric_dtype similar to pandas for most cases
        if pd.api.types.is_numeric_dtype(df[c].to_pandas() if isinstance(df, cudf.DataFrame) else df[c]):
            # fillna
            ser = df[c]
            mean_val = float(ser.mean())
            ser_filled = ser.fillna(mean_val).astype('float32')
            parts.append(ser_filled.rename(c))
        else:
            dummies = cudf.get_dummies(df[c].astype('str'), prefix=c).drop(columns=[f"{c}_{df[c].astype('category').cat.categories[0]}"], errors='ignore') if False else cudf.get_dummies(df[c].astype('str'), prefix=c)
            # drop first column to mimic drop_first behavior
            if dummies.shape[1] > 0:
                dummies = dummies.iloc[:, 1:]
            parts.append(dummies)
    if not parts:
        raise ValueError("No valid matching features")
    X = cudf.concat(parts, axis=1)
    # Standardize columns (columnwise)
    # Convert to float32
    X = X.astype('float32')
    # mean/std per column
    for col in X.columns:
        col_mean = float(X[col].mean())
        col_std = float(X[col].std())
        if col_std == 0:
            col_std = 1.0
        X[col] = (X[col] - col_mean) / col_std
    return X

def match_control_units_gpu(df: pd.DataFrame, gene_col: str, k: int, covariates_for_matching: List[str]):
    """
    GPU matching using cudf + cuML.NearestNeighbors.
    Input df is pandas, converted to cudf inside.
    Returns matched pandas DataFrame (treated + selected controls).
    """
    if not RAPIDS_AVAILABLE:
        raise RuntimeError("RAPIDS (cudf/cuml) not available")

    # convert to cudf
    gdf = cudf.DataFrame.from_pandas(df.reset_index(drop=True))
    treated = gdf[gdf[gene_col] == 1]
    control = gdf[gdf[gene_col] == 0]
    if len(treated) == 0 or len(control) == 0:
        return None

    gdf_all = cudf.concat([treated, control], ignore_index=True)
    # prepare X as cudf DF
    X = prepare_matching_matrix_cudf(gdf_all, covariates_for_matching)
    # create mask for treated
    mask_t = (gdf_all[gene_col] == 1).to_array() if hasattr(gdf_all[gene_col], "to_array") else (gdf_all[gene_col].to_pandas().values == 1)
    # convert to cupy arrays for cuML
    try:
        import cupy as cp
        X_cp = cp.asarray(X.to_pandas().values) if isinstance(X, cudf.DataFrame) else cp.asarray(X)
    except Exception:
        # cuML accepts cudf DataFrame directly in many versions
        X_cp = X

    # split X
    # number treated
    n_t = int((gdf_all[gene_col] == 1).sum())
    X_t = X.iloc[:n_t, :] if isinstance(X, cudf.DataFrame) else X[:n_t]
    X_c = X.iloc[n_t:, :] if isinstance(X, cudf.DataFrame) else X[n_t:]

    # Fit cuML NearestNeighbors on controls
    knn = cuNearestNeighbors(n_neighbors=min(k, len(X_c)), algorithm="brute", metric="euclidean")
    knn.fit(X_c)
    distances, indices = knn.kneighbors(X_t)
    # indices may be GPU arrays; convert to numpy
    try:
        idx_np = indices.get() if hasattr(indices, "get") else indices
    except Exception:
        try:
            import cupy as cp
            idx_np = cp.asnumpy(indices)
        except Exception:
            idx_np = np.array(indices)

    # flatten and add offset to get indices within gdf_all (controls start at n_t)
    selected = np.unique(idx_np.flatten())
    ctrl_idx = (selected + n_t).astype(int)
    # collect matched
    # Convert gdf_all back to pandas for slicing
    pdf_all = gdf_all.to_pandas()
    matched_controls = pdf_all.iloc[ctrl_idx]
    matched_treated = pdf_all.iloc[:n_t]
    matched_df = pd.concat([matched_treated, matched_controls], ignore_index=True)
    return matched_df

def match_control_units_cpu(df: pd.DataFrame, gene_col: str, k: int, covariates_for_matching: List[str]):
    """
    CPU fallback matching using sklearn.NearestNeighbors.
    Returns matched pandas DataFrame.
    """
    X, colsX = prepare_matching_matrix_pandas(df, covariates_for_matching)
    # mask split
    mask_t = (df[gene_col] == 1).values
    X_t = X[mask_t, :]
    X_c = X[~mask_t, :]
    if X_c.shape[0] == 0 or X_t.shape[0] == 0:
        return None
    k_used = min(k, X_c.shape[0])
    nn = skNearestNeighbors(n_neighbors=k_used)
    nn.fit(X_c)
    indices = nn.kneighbors(X_t, return_distance=False)
    selected = np.unique(indices.flatten())
    # map selected back to original indices in df
    control_indices = np.where(~mask_t)[0]
    ctrl_idx = control_indices[selected]
    treated_idx = np.where(mask_t)[0]
    matched_controls = df.iloc[ctrl_idx]
    matched_treated = df.iloc[treated_idx]
    return pd.concat([matched_treated, matched_controls], ignore_index=True)

def match_control_units(df: pd.DataFrame, gene_col: str, k: int, covariates_for_matching: List[str]):
    """
    Public wrapper: prefer RAPIDS GPU matching, else sklearn CPU matching.
    """
    if RAPIDS_AVAILABLE:
        try:
            return match_control_units_gpu(df, gene_col, k, covariates_for_matching)
        except Exception as e:
            warn(f"GPU matching failed with error: {e}. Falling back to CPU matching.")
    if SKLEARN_AVAILABLE:
        return match_control_units_cpu(df, gene_col, k, covariates_for_matching)
    raise RuntimeError("No matching backend available (install cuml or scikit-learn).")

# ----------------- LINEAR MODEL ON GPU (PyTorch) -----------------

def build_design_matrix_torch(df: pd.DataFrame, gene_col: str, Ecols: List[str], covariates: List[str]):
    """
    Build design matrix X (torch tensor on device) following:
    columns order:
      [ gene,
        exposures...,
        interaction gene*exposure... (same order as exposures),
        covariates numeric,
        covariate dummies...
      ]
    No intercept included (we'll add later in linear_regression).
    """
    tensors = []
    # gene
    gene_arr = torch.tensor(df[gene_col].astype(float).values, dtype=torch.float32, device=device).unsqueeze(1)
    tensors.append(gene_arr)
    # exposures and interactions
    for e in Ecols:
        arr_e = torch.tensor(df[e].astype(float).fillna(df[e].mean()).values, dtype=torch.float32, device=device).unsqueeze(1)
        tensors.append(arr_e)
        tensors.append(arr_e * gene_arr)
    # covariates
    for c in covariates:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            arr_c = torch.tensor(df[c].fillna(df[c].mean()).astype(float).values, dtype=torch.float32, device=device).unsqueeze(1)
            tensors.append(arr_c)
        else:
            # one-hot with pandas and then move to torch (drop first)
            dummies = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
            if dummies.shape[1] > 0:
                arr = torch.tensor(dummies.values.astype(np.float32), dtype=torch.float32, device=device)
                tensors.append(arr)
    if not tensors:
        raise ValueError("No columns for design matrix")
    X = torch.cat(tensors, dim=1)
    return X

def linear_regression_torch(X: torch.Tensor, y: torch.Tensor):
    """
    Solve least squares on GPU:
    We add intercept column inside this function.
    Returns beta tensor (p+1,) corresponding to intercept + coefficients.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for GPU linear regression")
    n = X.shape[0]
    intercept = torch.ones((n, 1), dtype=torch.float32, device=device)
    Xfull = torch.cat([intercept, X], dim=1)  # [n, p+1]
    y = y.unsqueeze(1)  # [n,1]
    # Use lstsq or pseudo-inverse depending on torch version
    try:
        # torch.linalg.lstsq introduced in newer versions
        sol = torch.linalg.lstsq(Xfull, y).solution  # [p+1,1]
        beta = sol.squeeze(1)
    except Exception:
        # fallback: normal equations with regularization (ridge small)
        XtX = Xfull.t().mm(Xfull)
        lam = 1e-6
        XtX_reg = XtX + lam * torch.eye(XtX.shape[0], device=device)
        Xty = Xfull.t().mm(y)
        beta = torch.linalg.solve(XtX_reg, Xty).squeeze(1)
    return beta  # 1D tensor length p+1

# ----------------- PROCESS SINGLE GENE -----------------

def _find_interaction_index(Ecols: List[str]) -> int:
    """
    Given layout in build_design_matrix_torch, the first interaction coefficient index (0-based
    for X columns, but remember intercept in betas) is:
    gene col is col 0, exposures cols follow; interactions start at column index 1 + len(Ecols)
    in the Xfull with intercept removed. To simplify: compute index of first interaction in beta vector:
    beta layout: [intercept, gene, exp1, gene*exp1, exp2, gene*exp2, ...]
    So the first interaction is at position: 1 + 1 + 1 = 3? We'll compute programmatically in process_single_gene.
    """
    return None  # placeholder (we compute exact index in process_single_gene)

def process_single_gene_gpu(gene_col: str, gene_original: str, Ecols: List[str]):
    """
    Full process for single gene using GPU for numeric ops (matching may use cuML).
    """
    info(f"[START] {gene_original}")

    # Load shared dataframe (pandas) from pickle (df with safe gene names)
    df = pickle.load(open(TEMP_DF_PATH, "rb"))

    conn = get_conn()

    # Skip if already done
    if conn is not None and gene_already_done(conn, gene_original):
        info(f"[SKIP DB] {gene_original}")
        if conn is not None:
            try: conn.close()
            except: pass
        return None

    # counts
    n_treated = int((df[gene_col] == 1).sum())
    n_control = int((df[gene_col] == 0).sum())
    if n_treated < min_treated or n_control == 0:
        info(f"[SKIP] Too few treated ({n_treated}) or controls ({n_control}) for {gene_original}")
        if conn is not None:
            try: conn.close()
            except: pass
        return None

    # prepare modeling dataframe (drop rows with missing in required cols)
    cols_needed = [onset_col, gene_col] + Ecols + covariates
    cols_present = [c for c in cols_needed if c in df.columns]
    df_model = df[cols_present].dropna()
    if df_model.shape[0] < min_sample_size:
        info(f"[SKIP] Not enough complete cases for {gene_original}: {df_model.shape[0]}")
        if conn is not None:
            try: conn.close()
            except: pass
        return None

    # Matching on covariates (Ecols + covariates)
    cov_match = [c for c in (Ecols + covariates) if c in df_model.columns]
    try:
        matched_obs = match_control_units(df_model, gene_col, k=match_k, covariates_for_matching=cov_match)
    except Exception as e:
        warn(f"[MATCH ERROR] {gene_original}: {e}")
        if conn is not None:
            try: conn.close()
            except: pass
        return None

    if matched_obs is None or matched_obs.shape[0] < min_sample_size:
        info(f"[SKIP] Matching failed or too small for {gene_original}")
        if conn is not None:
            try: conn.close()
            except: pass
        return None

    # Diagnostic: check simple SMD (on CPU for simplicity)
    try:
        smd_vals = compute_smd_basic(matched_obs, gene_col, cov_match)
        max_smd = max(smd_vals.values()) if len(smd_vals) > 0 else 0.0
        info(f"[{gene_original}] Max SMD dopo matching: {max_smd:.3f}")
        if max_smd > 0.25:
            warn(f"{gene_original} max SMD {max_smd:.3f} > 0.25")
    except Exception as ex:
        warn(f"SMD calc failed: {ex}")

    # Build design matrix (torch) and response
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for GPU regression")

    X_obs = build_design_matrix_torch(matched_obs, gene_col, Ecols, covariates)
    y_obs = torch.tensor(matched_obs[onset_col].astype(float).values, dtype=torch.float32, device=device)

    beta_obs = linear_regression_torch(X_obs, y_obs)  # includes intercept at beta_obs[0]

    # Find index for first interaction in beta vector:
    # beta layout: [intercept, gene, exp1, gene*exp1, exp2, gene*exp2, ... , covariates...]
    # So interaction for exp_i is at: idx = 1 + 1 + (i*2) + 1? Simpler to compute programmatically:
    # Let's reconstruct columns count to locate the first interaction coefficient index.
    # We built X with order: gene (1), for each exp: exp, gene*exp, then covariates...
    num_gene = 1
    num_exps = len([e for e in Ecols if e in matched_obs.columns])
    # index of first interaction in X columns (0-based): num_gene + 1 (skip first exp)?? compute properly:
    # X columns order (0-based): 0:gene, 1:exp1, 2:gene*exp1, 3:exp2,4:gene*exp2,...
    # So the first interaction is at pos 2 (if at least 1 exposure)
    if num_exps < 1:
        warn("No exposures found in matched_obs; skipping gene.")
        if conn is not None:
            try: conn.close()
            except: pass
        return None
    first_interaction_pos_in_X = 2  # 0-based in X
    # beta vector has intercept prepended, so add +1
    first_interaction_beta_idx = 1 + first_interaction_pos_in_X

    obs_coef = float(beta_obs[first_interaction_beta_idx].item())

    # PERMUTATION TEST (fully GPU for tensor ops; matching uses cuML if available)
    perm_betas = []
    # prepare original gene tensor on device for permutation
    gene_orig_full = torch.tensor(df_model[gene_col].astype(float).values, dtype=torch.float32, device=device)

    rng_state = random_state + abs(hash(gene_col)) % 2_000_000
    # Use torch.Generator for reproducible permutations
    gen = torch.Generator(device=device)
    gen.manual_seed(rng_state)

    # iterate permutations
    for i in range(n_perm):
        # permute gene vector on GPU
        perm_idx = torch.randperm(gene_orig_full.size(0), generator=gen, device=device)
        permuted_gene = gene_orig_full[perm_idx].cpu().numpy()  # move to CPU for dataframe insertion

        # create permuted df_model copy (only gene column changed)
        df_perm = df_model.copy()
        df_perm[gene_col] = permuted_gene

        # match on permuted data
        try:
            matched_perm = match_control_units(df_perm, gene_col, k=match_k, covariates_for_matching=cov_match)
            if matched_perm is None or matched_perm.shape[0] < min_sample_size:
                perm_betas.append(np.nan)
                continue
            X_perm = build_design_matrix_torch(matched_perm, gene_col, Ecols, covariates)
            y_perm = torch.tensor(matched_perm[onset_col].astype(float).values, dtype=torch.float32, device=device)
            beta_perm = linear_regression_torch(X_perm, y_perm)
            perm_betas.append(float(beta_perm[first_interaction_beta_idx].item()))
        except Exception as ex:
            perm_betas.append(np.nan)

    perm_betas = np.array([x for x in perm_betas if not np.isnan(x)])
    p_emp = float(np.mean(np.abs(perm_betas) >= np.abs(obs_coef))) if perm_betas.size > 0 else None

    # Save results
    try:
        save_gene_result(conn, gene_original,
                         int((matched_obs[gene_col] == 1).sum()),
                         int((matched_obs[gene_col] == 0).sum()),
                         obs_coef,
                         float(np.mean(perm_betas)) if perm_betas.size > 0 else None,
                         float(np.std(perm_betas)) if perm_betas.size > 0 else None,
                         p_emp)
    except Exception as ex:
        warn(f"Save failed for {gene_original}: {ex}")

    info(f"[DONE] {gene_original} (p_emp={p_emp})")
    if conn is not None:
        try: conn.close()
        except: pass
    return gene_original

# ----------------- SMD diagnostic (simple CPU implementation) -----------------
def compute_smd_basic(matched_df: pd.DataFrame, gene_col: str, covs: List[str]):
    """
    Basic SMD calculation on CPU for numeric and categorical (via dummies).
    Returns dict cov->smd
    """
    res = {}
    treated = matched_df[matched_df[gene_col] == 1]
    control = matched_df[matched_df[gene_col] == 0]
    for c in covs:
        if c not in matched_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(matched_df[c]):
            t = treated[c].fillna(treated[c].mean())
            u = control[c].fillna(control[c].mean())
            mean_t = t.mean(); mean_c = u.mean()
            std_t = t.std(ddof=1); std_c = u.std(ddof=1)
            n_t = t.count(); n_c = u.count()
            if n_t + n_c - 2 > 0:
                pooled = math.sqrt(((n_t-1)*std_t*std_t + (n_c-1)*std_c*std_c) / (n_t + n_c - 2))
            else:
                pooled = 1.0
            smd = 0.0 if pooled == 0 else abs(mean_t - mean_c) / pooled
            res[c] = smd
        else:
            dummies = pd.get_dummies(matched_df[c].astype(str), prefix=c, drop_first=True)
            for col in dummies.columns:
                dt = dummies.loc[treated.index, col]
                dc = dummies.loc[control.index, col]
                pt = dt.mean(); pc = dc.mean()
                p_pool = (pt * len(dt) + pc * len(dc)) / (len(dt) + len(dc))
                pooled = math.sqrt(p_pool * (1 - p_pool)) if 0 < p_pool < 1 else 1.0
                res[col] = 0.0 if pooled == 0 else abs(pt - pc) / pooled
    return res

# ----------------- MULTIPLE TEST CORRECTION & PLOT -----------------

def add_fdr(df_results: pd.DataFrame, p_col="empirical_p", fdr_col="fdr"):
    df = df_results.copy()
    if p_col not in df.columns:
        raise ValueError(f"{p_col} not in results")
    pvals = df[p_col].astype(float).fillna(1.0).values
    df[fdr_col] = multipletests(pvals, method="fdr_bh")[1]
    return df

def volcano_plot(df, beta_col="obs_coef", p_col="empirical_p", fdr_col="fdr", save_path="volcano_plot.png"):
    df2 = df.copy()
    df2["neglog10p"] = -np.log10(df2[p_col].astype(float).replace(0, np.nextafter(0,1)))
    plt.figure(figsize=(10,7))
    plt.scatter(df2[beta_col], df2["neglog10p"], alpha=0.6)
    sig = df2[df2[fdr_col] < 0.05] if fdr_col in df2.columns else df2[df2[p_col].astype(float) < 0.05]
    if not sig.empty:
        plt.scatter(sig[beta_col], sig["neglog10p"], s=50, edgecolor='k', label='significant', color='orange')
    plt.xlabel("Interaction beta")
    plt.ylabel("-log10(p)")
    plt.title("Volcano plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    info(f"Volcano plot saved to {save_path}")

# ----------------- MAIN -----------------

def main():
    info("Starting GPU pipeline")
    # Preflight hardware / library checks
    info(f"TORCH_AVAILABLE={TORCH_AVAILABLE}, RAPIDS_AVAILABLE={RAPIDS_AVAILABLE}, SKLEARN_AVAILABLE={SKLEARN_AVAILABLE}")
    if TORCH_AVAILABLE and torch.cuda.is_available():
        info(f"Using device: {device}")
    elif TORCH_AVAILABLE:
        warn("PyTorch available but CUDA not available; running on CPU with torch.")
    else:
        warn("PyTorch not available; the pipeline requires PyTorch for GPU regression.")

    # Load genetic data
    df_gen = pd.read_csv(raw_file, sep=sep, decimal=decimal)
    non_gen_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]
    # drop columns with too many -1
    mask = (df_gen == -1).mean() < 0.30
    df_gen = df_gen.loc[:, mask.values]
    gene_cols = [c for c in df_gen.columns if c not in non_gen_cols]
    for g in gene_cols:
        df_gen[g] = (df_gen[g] > 0).astype(int)
    if "IID" in df_gen.columns and "id" not in df_gen.columns:
        df_gen = df_gen.rename(columns={"IID": "id"})

    # Load environment data
    df_env = pd.read_csv(env_file, sep=sep, decimal=decimal)
    # ensure categorical dtypes for categorical covariates
    for c in ["sex", "onset_site"]:
        if c in df_env.columns:
            df_env[c] = df_env[c].astype("category")

    # Merge
    df = pd.merge(df_env, df_gen, on="id", how="inner")
    df[onset_col] = pd.to_numeric(df[onset_col], errors="coerce")

    # Standardize exposures (we'll keep pandas columns and also rely on torch later)
    Ecols = []
    for exp in exposures:
        if exp not in df.columns:
            warn(f"Exposure {exp} not found in env file")
            continue
        df[exp] = pd.to_numeric(df[exp], errors="coerce")
        if standardize:
            scaler = StandardScaler()
            df[exp + "_std"] = scaler.fit_transform(df[[exp]]).astype(np.float32)
            Ecols.append(exp + "_std")
        else:
            Ecols.append(exp)

    # Safe gene names (map original -> gene_i)
    safe_map = {g: f"gene_{i}" for i, g in enumerate(gene_cols)}
    df = df.rename(columns=safe_map)
    gene_cols_safe = list(safe_map.values())
    mapping_safe_to_original = {v: k for k, v in safe_map.items()}

    info(f"Total variants: {len(gene_cols_safe)}")

    # Save pickle for worker processes / reuse
    df.to_pickle(TEMP_DF_PATH)

    # Sequential processing (GPU pipeline often faster sequentially due to single GPU resource).
    results = []
    for gc in tqdm(gene_cols_safe, desc="Genes"):
        try:
            res = process_single_gene_gpu(gc, mapping_safe_to_original[gc], Ecols)
            results.append(res)
        except Exception as e:
            warn(f"Error processing {gc}: {e}")

    # Load results table from DB (if available) or prepare toy results
    try:
        results_df = load_gene_results()
    except Exception as e:
        warn(f"load_gene_results failed: {e}")
        results_df = pd.DataFrame()

    # If results_df is empty, create minimal placeholder from saved results (if save_gene_result printed)
    if results_df is None or results_df.empty:
        info("No DB results found; results_df empty.")
    else:
        results_df = add_fdr(results_df)
        volcano_plot(results_df, save_path="volcano_plot.png")
    info("Pipeline finished.")

if __name__ == "__main__":
    main()
