import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf
from concurrent.futures import ProcessPoolExecutor, as_completed
from db import gene_already_done, load_gene_results, save_gene_result
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import os

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
max_workers = 16
TEMP_DF_PATH = "temp_df.pkl"
match_k = 3
# ----------------------------------------

np.random.seed(random_state)
random.seed(random_state)

# ---------------- FORMULA ----------------
def build_formula(onset_col, gene_col, exposures, covariates, df_subset):
    exposures_str = " + ".join(exposures)
    formula = f"{onset_col} ~ {gene_col} * ({exposures_str})"
    cov_in_df = [c for c in covariates if c in df_subset.columns]
    if cov_in_df:
        formula += " + " + " + ".join(cov_in_df)
    return formula

# ---------------- MATCHING ----------------
def _prepare_matching_matrix(df, cols):
    features = []
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(df[c].fillna(df[c].mean()))
        else:
            features.append(pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True))
    if not features:
        raise ValueError("No valid matching features found")
    X = pd.concat(features, axis=1)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled

def match_control_units(df, gene_col, k=2, covariates_for_matching=None):
    treated = df[df[gene_col] == 1].reset_index(drop=True)
    control = df[df[gene_col] == 0].reset_index(drop=True)
    if treated.shape[0] == 0 or control.shape[0] == 0:
        return None
    df_matching = pd.concat([treated, control], ignore_index=True)
    X = _prepare_matching_matrix(df_matching, covariates_for_matching)
    mask_t = df_matching[gene_col] == 1
    X_t = X[mask_t]
    X_c = X[~mask_t]
    if X_c.shape[0] == 0:
        return None
    k_used = min(k, X_c.shape[0])
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_c.values)
    _, indices = nn.kneighbors(X_t.values)
    ctrl_idx = X_c.index[np.unique(indices.flatten())]
    matched_controls = df_matching.loc[ctrl_idx]
    matched_treated = df_matching.loc[mask_t]
    return pd.concat([matched_treated, matched_controls], ignore_index=True)

# ---------------- HELPER ----------------
def _find_interaction_term(params_index, gene_col):
    for name in params_index:
        if ":" in name and gene_col in name:
            return name
    return None

# ---------------- PERMUTATION ----------------
def permutation_test_interaction(df_original, gene_col, formula_builder,
                                 onset_col, Ecols, covariates, match_k, n_perm=1000, seed=42):
    rng = np.random.RandomState(seed)
    matched_obs = match_control_units(df_original, gene_col, k=match_k, covariates_for_matching=Ecols+covariates)
    if matched_obs is None or matched_obs.shape[0]<5:
        return None, None, np.array([])
    formula_obs = formula_builder(onset_col, gene_col, Ecols, covariates, matched_obs)
    mod_obs = smf.ols(formula=formula_obs, data=matched_obs).fit()
    interaction_name = _find_interaction_term(mod_obs.params.index, gene_col)
    if interaction_name is None:
        return None, None, np.array([])
    beta_obs = mod_obs.params[interaction_name]

    perm_betas = []
    for _ in range(n_perm):
        df_perm = df_original.copy()
        df_perm[gene_col] = rng.permutation(df_perm[gene_col].values)
        matched_perm = match_control_units(df_perm, gene_col, k=match_k, covariates_for_matching=Ecols+covariates)
        if matched_perm is None or matched_perm.shape[0]<5:
            perm_betas.append(np.nan)
            continue
        try:
            formula_perm = formula_builder(onset_col, gene_col, Ecols, covariates, matched_perm)
            mod_perm = smf.ols(formula=formula_perm, data=matched_perm).fit()
            perm_betas.append(mod_perm.params.get(interaction_name, np.nan))
        except:
            perm_betas.append(np.nan)
    perm_betas = np.array([x for x in perm_betas if not np.isnan(x)])
    p_emp = float(np.mean(np.abs(perm_betas) >= np.abs(beta_obs))) if perm_betas.size>0 else None
    return float(beta_obs), p_emp, perm_betas

# ---------------- PROCESS SINGLE GENE ----------------
def process_single_gene(gene_col, gene_original, Ecols):
    df = pickle.load(open(TEMP_DF_PATH, "rb"))
    if gene_already_done(gene_original):
        return None
    n_treated = int((df[gene_col]==1).sum())
    n_control = int((df[gene_col]==0).sum())
    if n_treated<min_treated or n_control==0:
        return None
    cols_model = [onset_col, gene_col]+Ecols+covariates
    df_model = df[cols_model].dropna()
    if df_model.shape[0]<min_sample_size:
        return None
    matched = match_control_units(df_model, gene_col, k=match_k, covariates_for_matching=Ecols+covariates)
    if matched is None or matched.shape[0]<min_sample_size:
        return None
    formula = build_formula(onset_col, gene_col, Ecols, covariates, matched)
    try:
        mod = smf.ols(formula=formula, data=matched).fit()
        interaction_name = _find_interaction_term(mod.params.index, gene_col)
        obs_coef = float(mod.params[interaction_name]) if interaction_name else None
    except:
        obs_coef = None
    beta_obs, p_emp, perm_coefs = permutation_test_interaction(df, gene_col, build_formula,
                                                               onset_col, Ecols, covariates, match_k, n_perm=n_perm)
    save_gene_result(gene_original,
                     int(matched[gene_col].sum()),
                     int((matched[gene_col]==0).sum()),
                     obs_coef,
                     float(np.mean(perm_coefs)) if perm_coefs.size>0 else None,
                     float(np.std(perm_coefs)) if perm_coefs.size>0 else None,
                     p_emp)
    return gene_original

# ---------------- MAIN ----------------
def main():
    df_gen = pd.read_csv(raw_file, sep=sep, decimal=decimal)
    df_gen = df_gen.loc[:, (df_gen == -1).mean()<0.3]
    non_gen_cols = ["FID","IID","PAT","MAT","SEX","PHENOTYPE","id"]
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

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_single_gene, gc, mapping[gc], Ecols) for gc in gene_cols_safe]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("Errore in un processo:", e)

    results_df = load_gene_results()
    results_df["fdr"] = multipletests(results_df["empirical_p"], method="fdr_bh")[1]

    results_df["neglog10p"] = -np.log10(results_df["empirical_p"])
    plt.scatter(results_df["obs_coef"], results_df["neglog10p"], alpha=0.6)
    sig_fdr = results_df[results_df["fdr"]<0.05]
    plt.scatter(sig_fdr["obs_coef"], sig_fdr["neglog10p"], color="orange", edgecolor="black")
    plt.xlabel("Beta interazione"); plt.ylabel("-log10(p)")
    plt.title("Volcano Plot Gene × Ambiente")
    plt.savefig("volcano_plot.png", dpi=300)
    plt.close()

if __name__=="__main__":
    main()
