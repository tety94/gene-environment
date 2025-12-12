import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf
from concurrent.futures import ProcessPoolExecutor, as_completed
from db import gene_already_done, load_gene_results, save_gene_result, get_conn
from tqdm import tqdm
import numpy as np
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
# number of permutations for the permutation test
n_perm = 1000
random_state = 42
standardize = True
min_treated = 5
min_sample_size = 10
max_workers = 16
TEMP_DF_PATH = "temp_df.pkl"
# number of matched controls per treated unit
match_k = 3
# ----------------------------------------

np.random.seed(random_state)
random.seed(random_state)


def build_formula(onset_col, gene_col, exposures, covariates, df_subset):
    exposures_str = " + ".join(exposures)
    formula = f"{onset_col} ~ {gene_col} * ({exposures_str})"
    cov_in_df = [c for c in covariates if c in df_subset.columns]
    if cov_in_df:
        formula += " + " + " + ".join(cov_in_df)
    return formula


# ----------------- HELPERS PER MATCHING -----------------
def _prepare_matching_matrix(df, cols):
    """Return a numeric matrix suitable for NearestNeighbors.
    - numeric columns are used as-is
    - categorical columns are one-hot encoded
    Missing values are imputed with column mean.
    """
    if not cols:
        raise ValueError("No columns provided for matching")

    mat = []
    df_copy = df.copy()

    # build a DataFrame with numeric features
    features = []
    for c in cols:
        if c not in df_copy.columns:
            continue
        if pd.api.types.is_numeric_dtype(df_copy[c]):
            features.append(df_copy[c].fillna(df_copy[c].mean()))
        else:
            # one-hot encode, drop first to avoid collinearity
            dummies = pd.get_dummies(df_copy[c].astype(str), prefix=c, drop_first=True)
            features.append(dummies)

    if not features:
        raise ValueError("No valid matching features found in dataframe")

    X = pd.concat(features, axis=1)
    # scale features to unit variance (important for NN)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled


def match_control_units(df, gene_col, k=2, covariates_for_matching=None):
    """
    Nearest-neighbor matching: for each treated unit (gene==1) find k nearest controls (gene==0)
    Returns a matched dataframe containing all treated and the selected controls.
    """
    if covariates_for_matching is None:
        raise ValueError("Serve almeno una covariata per il matching")

    treated = df[df[gene_col] == 1].reset_index(drop=True)
    control = df[df[gene_col] == 0].reset_index(drop=True)

    if treated.shape[0] == 0 or control.shape[0] == 0:
        return None

    # prepare matrices
    df_matching = pd.concat([treated, control], ignore_index=True)
    X = _prepare_matching_matrix(df_matching, covariates_for_matching)

    # split back
    X_t = X.loc[treated.index]
    X_c = X.loc[control.index + len(treated)] if len(control) > 0 else pd.DataFrame()

    # in case indices are not aligned as expected, safer approach:
    # build masks
    mask_t = df_matching[gene_col] == 1
    X_t = X[mask_t]
    X_c = X[~mask_t]

    if X_c.shape[0] == 0:
        return None

    # adjust k if not enough controls
    k_used = min(k, X_c.shape[0])

    nn = NearestNeighbors(n_neighbors=k_used).fit(X_c.values)
    distances, indices = nn.kneighbors(X_t.values)

    # get indices in X_c corresponding to selected controls
    selected_ctrl_pos = np.unique(indices.flatten())
    ctrl_idx = X_c.index[selected_ctrl_pos]

    matched_controls = df_matching.loc[ctrl_idx]
    matched_treated = df_matching.loc[mask_t]

    matched_df = pd.concat([matched_treated, matched_controls], ignore_index=True)
    return matched_df


# ----------------- PERMUTATION TEST -----------------

def _find_interaction_term(mod_params_index, gene_col):
    """Find the first interaction parameter name that involves gene_col."""
    for name in mod_params_index:
        if ":" in name and gene_col in name:
            return name
        if ":" in name and any(part == gene_col for part in name.split(":")):
            return name
    return None

def process_single_gene(gene_col, gene_original, Ecols):

    print(f"[START] {gene_original}")

    # ogni processo legge il DF dal pickle
    df = pickle.load(open(TEMP_DF_PATH, "rb"))
    conn = get_conn()
    if gene_already_done(conn, gene_original):
        print(f"[SKIP] {gene_original}")
        return None

    # check counts
    n_treated = int((df[gene_col] == 1).sum())
    n_control = int((df[gene_col] == 0).sum())
    if n_treated < min_treated or n_control == 0:
        print(f"[SKIP] Too few treated ({n_treated}) or controls ({n_control}) for {gene_original}")
        return None

    # columns for modeling
    cols = [onset_col, gene_col] + Ecols + covariates
    df_model = df[cols].dropna()
    if df_model.shape[0] < min_sample_size:
        print(f"[SKIP] Not enough complete cases for {gene_original}: {df_model.shape[0]}")
        return None

    # ---------- MATCHING ----------
    cov_match = Ecols + covariates
    try:
        matched = match_control_units(df_model, gene_col, k=match_k, covariates_for_matching=cov_match)
    except Exception as e:
        print(f"[MATCH ERROR] {gene_original}: {e}")
        return None

    if matched is None or matched.shape[0] < min_sample_size:
        print(f"[SKIP] Matching failed or too small for {gene_original}")
        return None

    # ---------- FIT MODEL ON MATCHED SAMPLE (OBSERVED) ----------
    try:
        formula = build_formula(onset_col, gene_col, Ecols, covariates, matched)
        mod = smf.ols(formula=formula, data=matched).fit()
        interaction_name = _find_interaction_term(mod.params.index, gene_col)
        if interaction_name is None:
            print(f"[SKIP] No interaction term found for {gene_original}")
            return None

        obs_coef = float(mod.params[interaction_name])
    except Exception as e:
        print(f"[ERROR OBS] {gene_original}: {e}")
        obs_coef = None

    # ---------- PERMUTATION TEST DENTRO MATCHED SAMPLE ----------
    rng = np.random.RandomState(random_state + (abs(hash(gene_col)) % 2_000_000))
    perm_betas = []

    for _ in tqdm(range(n_perm), desc=f"Perm test {gene_col}", leave=False):
        df_perm = matched.copy()
        df_perm[gene_col] = rng.permutation(df_perm[gene_col].values)
        try:
            mod_perm = smf.ols(formula=formula, data=df_perm).fit()
            perm_betas.append(mod_perm.params.get(interaction_name, np.nan))
        except Exception:
            perm_betas.append(np.nan)

    perm_betas = np.array([x for x in perm_betas if not np.isnan(x)])
    p_emp = float(np.mean(np.abs(perm_betas) >= np.abs(obs_coef))) if perm_betas.size > 0 else None

    # ---------- SAVE RESULTS ----------
    try:
        save_gene_result(
            conn,
            gene_original,
            int(matched[gene_col].sum()),  # treated in matched
            int((matched[gene_col] == 0).sum()),  # controls matched
            obs_coef,
            float(np.mean(perm_betas)) if perm_betas.size > 0 else None,
            float(np.std(perm_betas)) if perm_betas.size > 0 else None,
            p_emp,
        )
    except Exception as e:
        print(f"[SAVE ERROR] {gene_original}: {e}")

    print(f"[DONE] {gene_original}")
    conn.close()
    return gene_original


# ------------ PROCESSA UN GENE (NEI PROCESSI PARALLELI) ------------
def permutation_test_interaction(
    df_original,
    gene_col,
    formula_builder,
    onset_col,
    Ecols,
    covariates,
    match_k,
    n_perm=1000,
    seed=42
):
    """
    Permutation test CORRETTO:
    - permuto il gene sull'intero dataset originale
    - rifaccio il matching per ogni permutazione
    - rifaccio il modello
    - raccolgo i beta permutati
    """

    rng = np.random.RandomState(seed)

    # ==== 1. Observed model sul matched vero ====
    matched_obs = match_control_units(
        df_original,
        gene_col,
        k=match_k,
        covariates_for_matching=Ecols + covariates
    )
    if matched_obs is None or matched_obs.shape[0] < 5:
        return None, None, np.array([])

    formula_obs = formula_builder(onset_col, gene_col, Ecols, covariates, matched_obs)
    mod_obs = smf.ols(formula=formula_obs, data=matched_obs).fit()

    interaction_name = _find_interaction_term(mod_obs.params.index, gene_col)
    if interaction_name is None:
        return None, None, np.array([])

    beta_obs = mod_obs.params[interaction_name]

    # ==== 2. Distribuzione nulla ====
    perm_betas = []

    for _ in tqdm(range(n_perm), desc=f"Perm test {gene_col}", leave=False):
        df_perm = df_original.copy()

        # permuto il gene
        df_perm[gene_col] = rng.permutation(df_perm[gene_col].values)

        # nuovo matching su permutato
        matched_perm = match_control_units(
            df_perm,
            gene_col,
            k=match_k,
            covariates_for_matching=Ecols + covariates
        )

        if matched_perm is None or matched_perm.shape[0] < 5:
            perm_betas.append(np.nan)
            continue

        # modello su permutato
        try:
            formula_perm = formula_builder(onset_col, gene_col, Ecols, covariates, matched_perm)
            mod_perm = smf.ols(formula=formula_perm, data=matched_perm).fit()
            beta_perm = mod_perm.params.get(interaction_name, np.nan)
        except Exception:
            beta_perm = np.nan

        perm_betas.append(beta_perm)

    perm_betas = np.array([x for x in perm_betas if not np.isnan(x)])

    if perm_betas.size == 0:
        return beta_obs, None, np.array([])

    # ==== p-value empirico ====
    p_emp = float(np.mean(np.abs(perm_betas) >= np.abs(beta_obs)))


    return float(beta_obs), p_emp, perm_betas


# ------------ MAIN SCRIPT ---------------
def main():

    # --------- LOAD GENETIC ---------
    df_gen = pd.read_csv(raw_file, sep=sep, decimal=decimal)
    non_gen_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]
    # drop columns with too many missing "-1"
    df_gen = df_gen.loc[:, (df_gen == -1).mean() < 0.30]
    gene_cols = [c for c in df_gen.columns if c not in non_gen_cols]

    # PER TEST
    # gene_cols = gene_cols[:5]

    for g in gene_cols:
        df_gen[g] = (df_gen[g] > 0).astype(int)


    if "IID" in df_gen.columns:
        df_gen = df_gen.rename(columns={"IID": "id"})

    # --------- LOAD ENV ---------
    df_env = pd.read_csv(env_file, sep=sep, decimal=decimal)
    df_env["sex"] = df_env["sex"].astype("category")
    df_env["onset_site"] = df_env["onset_site"].astype("category")

    df = pd.merge(df_env, df_gen, on="id", how="inner")
    df[onset_col] = pd.to_numeric(df[onset_col], errors="coerce")

    # --------- STANDARDIZE ---------
    Ecols = []
    for exp in exposures:
        df[exp] = pd.to_numeric(df[exp], errors="coerce")
        if standardize:
            df[exp + "_std"] = StandardScaler().fit_transform(df[[exp]])
            Ecols.append(exp + "_std")
        else:
            Ecols.append(exp)

    # --------- SAFE NAMES ---------
    safe = {g: f"gene_{i}" for i, g in enumerate(gene_cols)}
    df.rename(columns=safe, inplace=True)
    gene_cols_safe = list(safe.values())
    mapping = {v: k for k, v in safe.items()}

    print(f"Totale varianti: {len(gene_cols_safe)}")

    # --------- SALVA DF IN PICKLE PER I PROCESSI ---------
    df.to_pickle(TEMP_DF_PATH)

    # --------- PARALLEL PROCESSING ---------
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_single_gene, gc, mapping[gc], Ecols)
            for gc in gene_cols_safe
        ]

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("Errore in un processo:", e)

    results_df = load_gene_results()
    results_df = add_fdr(results_df)
    volcano_plot(results_df, save_path="volcano_plot.png")

def add_fdr(df, p_col="empirical_p", fdr_col="fdr"):
    df = df.copy()
    # converti in float
    pvals = df[p_col].astype(float).values
    df[fdr_col] = multipletests(pvals, method="fdr_bh")[1]
    return df


def volcano_plot(df,
                 beta_col="obs_coef",
                 p_col="empirical_p",
                 gene_col="gene",
                 p_thresh=1e-5,
                 fdr_col="fdr",
                 fdr_thresh=0.05,
                 save_path=None):
    df = df.copy()
    df["neglog10p"] = -np.log10(df[p_col])

    plt.figure(figsize=(9, 7))

    # tutti i punti
    plt.scatter(df[beta_col], df["neglog10p"], alpha=0.6)

    # linee cutoff
    plt.axhline(-np.log10(p_thresh), linestyle="--", color="red", label=f"p = {p_thresh}")

    # evidenzia FDR significativi
    if fdr_col in df.columns:
        sig_fdr = df[df[fdr_col] < fdr_thresh]
        plt.scatter(sig_fdr[beta_col], sig_fdr["neglog10p"],
                    s=50, edgecolor="black", label=f"FDR < {fdr_thresh}", color="orange")

    plt.xlabel("Beta dell'interazione")
    plt.ylabel("-log10(p)")
    plt.title("Volcano Plot: Interazioni Gene × Ambiente")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Volcano plot salvato in: {save_path}")
    else:
        plt.show()

    plt.close()

def load_or_compute_matched(df, gene_col, gene_original, k, covariates_for_matching):
    cache_file = f"matched_{gene_col}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            matched = pickle.load(f)
        print(f"[CACHE] Matching caricato per {gene_original}")
    else:
        matched = match_control_units(df, gene_col, k=k, covariates_for_matching=covariates_for_matching)
        with open(cache_file, "wb") as f:
            pickle.dump(matched, f)
        print(f"[CACHE] Matching salvato per {gene_original}")

    return matched

if __name__ == "__main__":
    main()