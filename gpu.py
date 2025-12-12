import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from concurrent.futures import ProcessPoolExecutor, as_completed
from db import gene_already_done, load_gene_results, save_gene_result, get_conn
from tqdm import tqdm
import torch
import os

# GPU cuML NearestNeighbors
try:
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
except ImportError:
    raise ImportError("Devi installare RAPIDS cuML per il matching su GPU")

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
# ----------------------------------------

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)


def build_formula(onset_col, gene_col, exposures, covariates, df_subset):
    exposures_str = " + ".join(exposures)
    formula = f"{onset_col} ~ {gene_col} * ({exposures_str})"
    cov_in_df = [c for c in covariates if c in df_subset.columns]
    if cov_in_df:
        formula += " + " + " + ".join(cov_in_df)
    return formula


def _prepare_matching_matrix(df, cols):
    """Return numeric matrix for cuML NN."""
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
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled.astype(np.float32)


def match_control_units(df, gene_col, k=2, covariates_for_matching=None, device=0):
    treated = df[df[gene_col] == 1].reset_index(drop=True)
    control = df[df[gene_col] == 0].reset_index(drop=True)
    if treated.shape[0] == 0 or control.shape[0] == 0:
        return None

    df_matching = pd.concat([treated, control], ignore_index=True)
    X = _prepare_matching_matrix(df_matching, covariates_for_matching)

    mask_t = df_matching[gene_col] == 1
    X_t = X[mask_t]
    X_c = X[~mask_t]

    k_used = min(k, X_c.shape[0])
    nn = cuNearestNeighbors(n_neighbors=k_used)
    nn.fit(X_c.to_numpy(), convert_dtype=True)

    distances, indices = nn.kneighbors(X_t.to_numpy())
    selected_ctrl_pos = np.unique(indices.flatten())
    ctrl_idx = X_c.index[selected_ctrl_pos]

    matched_controls = df_matching.loc[ctrl_idx]
    matched_treated = df_matching.loc[mask_t]
    matched_df = pd.concat([matched_treated, matched_controls], ignore_index=True)
    return matched_df


def check_balance(matched_df, gene_col, covariates_for_matching):
    smd_results = {}
    if matched_df is None:
        return smd_results
    treated = matched_df[matched_df[gene_col] == 1]
    control = matched_df[matched_df[gene_col] == 0]

    for c in covariates_for_matching:
        if c not in matched_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(matched_df[c]):
            c_t = treated[c].fillna(treated[c].mean())
            c_c = control[c].fillna(control[c].mean())
            pooled_std = np.sqrt(((len(c_t)-1)*c_t.std()**2 + (len(c_c)-1)*c_c.std()**2)/(len(c_t)+len(c_c)-2))
            smd_results[c] = np.abs(c_t.mean() - c_c.mean()) / (pooled_std if pooled_std>0 else 1)
        else:
            dummies = pd.get_dummies(matched_df[c].astype(str), prefix=c, drop_first=True)
            for d_col in dummies.columns:
                p_t = dummies.loc[treated.index, d_col].mean()
                p_c = dummies.loc[control.index, d_col].mean()
                pooled_std = np.sqrt(((p_t*(1-p_t)*len(treated)+p_c*(1-p_c)*len(control))/(len(treated)+len(control))))
                smd_results[d_col] = np.abs(p_t - p_c) / (pooled_std if pooled_std>0 else 1)
    return smd_results


def _find_interaction_term(mod_params_index, gene_col):
    for name in mod_params_index:
        if ":" in name and gene_col in name:
            return name
    return None


def process_single_gene(gene_col, gene_original, Ecols, device=0):
    print(f"[START] {gene_original} on GPU {device}")
    df = pickle.load(open(TEMP_DF_PATH, "rb"))
    conn = get_conn()
    if gene_already_done(conn, gene_original):
        print(f"[SKIP] {gene_original}")
        conn.close()
        return

    n_treated = int((df[gene_col] == 1).sum())
    n_control = int((df[gene_col] == 0).sum())
    if n_treated < min_treated or n_control == 0:
        print(f"[SKIP] Too few treated ({n_treated}) or controls ({n_control}) for {gene_original}")
        conn.close()
        return

    cols = [onset_col, gene_col] + Ecols + covariates
    df_model = df[cols].dropna()
    if df_model.shape[0] < min_sample_size:
        print(f"[SKIP] Not enough complete cases for {gene_original}")
        conn.close()
        return

    cov_match = Ecols + covariates
    try:
        matched_obs = match_control_units(df_model, gene_col, k=match_k, covariates_for_matching=cov_match, device=device)
    except Exception as e:
        print(f"[MATCH ERROR] {gene_original}: {e}")
        conn.close()
        return

    if matched_obs is None or matched_obs.shape[0] < min_sample_size:
        print(f"[SKIP] Matching failed for {gene_original}")
        conn.close()
        return

    smd_results = check_balance(matched_obs, gene_col, cov_match)
    max_smd = max(smd_results.values(), default=0)
    print(f"[{gene_original}] Max SMD dopo matching: {max_smd:.3f}")

    formula = build_formula(onset_col, gene_col, Ecols, covariates, matched_obs)
    try:
        mod = smf.ols(formula=formula, data=matched_obs).fit()
        interaction_name = _find_interaction_term(mod.params.index, gene_col)
        obs_coef = float(mod.params[interaction_name]) if interaction_name else None
    except Exception as e:
        print(f"[ERROR OBS] {gene_original}: {e}")
        obs_coef = None

    rng = np.random.RandomState(random_state + (abs(hash(gene_col)) % 2_000_000))
    perm_betas = []
    for _ in tqdm(range(n_perm), desc=f"Perm test {gene_col}", leave=False):
        df_perm = df_model.copy()
        df_perm[gene_col] = rng.permutation(df_perm[gene_col].values)
        try:
            matched_perm = match_control_units(df_perm, gene_col, k=match_k, covariates_for_matching=cov_match, device=device)
            if matched_perm is None or matched_perm.shape[0] < min_sample_size:
                perm_betas.append(np.nan)
                continue
            mod_perm = smf.ols(formula=formula, data=matched_perm).fit()
            perm_betas.append(mod_perm.params.get(interaction_name, np.nan))
        except Exception:
            perm_betas.append(np.nan)

    perm_betas = np.array([x for x in perm_betas if not np.isnan(x)])
    p_emp = float(np.mean(np.abs(perm_betas) >= np.abs(obs_coef))) if perm_betas.size > 0 else None

    try:
        save_gene_result(
            conn,
            gene_original,
            int(matched_obs[gene_col].sum()),
            int((matched_obs[gene_col] == 0).sum()),
            obs_coef,
            float(np.mean(perm_betas)) if perm_betas.size > 0 else None,
            float(np.std(perm_betas)) if perm_betas.size > 0 else None,
            p_emp,
        )
    except Exception as e:
        print(f"[SAVE ERROR] {gene_original}: {e}")

    print(f"[DONE] {gene_original}")
    conn.close()


def main():
    df_gen = pd.read_csv(raw_file, sep=sep, decimal=decimal)
    non_gen_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]
    df_gen = df_gen.loc[:, (df_gen == -1).mean() < 0.30]
    gene_cols = [c for c in df_gen.columns if c not in non_gen_cols]

    for g in gene_cols:
        df_gen[g] = (df_gen[g] > 0).astype(int)

    if "IID" in df_gen.columns:
        df_gen = df_gen.rename(columns={"IID": "id"})

    df_env = pd.read_csv(env_file, sep=sep, decimal=decimal)
    df_env["sex"] = df_env["sex"].astype("category")
    df_env["onset_site"] = df_env["onset_site"].astype("category")

    df = pd.merge(df_env, df_gen, on="id", how="inner")
    df[onset_col] = pd.to_numeric(df[onset_col], errors="coerce")

    Ecols = []
    for exp in exposures:
        df[exp] = pd.to_numeric(df[exp], errors="coerce")
        if standardize:
            df[exp + "_std"] = StandardScaler().fit_transform(df[[exp]])
            Ecols.append(exp + "_std")
        else:
            Ecols.append(exp)

    safe = {g: f"gene_{i}" for i, g in enumerate(gene_cols)}
    df.rename(columns=safe, inplace=True)
    gene_cols_safe = list(safe.values())
    mapping = {v: k for k, v in safe.items()}

    df.to_pickle(TEMP_DF_PATH)
    n_gpu = torch.cuda.device_count()
    print(f"{n_gpu} GPU disponibili")

    # dividiamo i geni tra le GPU
    chunks = np.array_split(gene_cols_safe, n_gpu)
    futures = []
    with ProcessPoolExecutor(max_workers=n_gpu) as ex:
        for i, chunk in enumerate(chunks):
            device = i  # GPU i-esima
            for gc in chunk:
                futures.append(ex.submit(process_single_gene, gc, mapping[gc], Ecols, device=device))
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("Errore in un processo:", e)


if __name__ == "__main__":
    main()
