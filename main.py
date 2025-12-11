import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf
from concurrent.futures import ProcessPoolExecutor, as_completed
from db import gene_already_done, save_gene_iteration, save_gene_result

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
max_workers = 12
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


def permutation_test_interaction(df, formula, gene_col, interaction_name, n_perm=5000, save_iter=False, gene_original=None, seed=42):
    """
    Performs a permutation test on the interaction coefficient. Optionally saves each permutation iteration via save_gene_iteration.

    Returns: (beta_obs, p_value, perm_coefs_array)
    """
    # observed model
    mod_obs = smf.ols(formula=formula, data=df).fit()
    beta_obs = mod_obs.params.get(interaction_name, np.nan)

    perm_coefs = []
    rng = np.random.RandomState(seed)

    for i in range(n_perm):
        df_perm = df.copy()
        df_perm[gene_col] = rng.permutation(df_perm[gene_col].values)

        try:
            mod_perm = smf.ols(formula=formula, data=df_perm).fit()
            perm_beta = mod_perm.params.get(interaction_name, np.nan)
        except Exception:
            perm_beta = np.nan

        perm_coefs.append(perm_beta)

        if save_iter and gene_original is not None:
            try:
                # p-value not meaningful per iteration; store coef and sample size
                save_gene_iteration(gene_original, i, float(perm_beta) if not np.isnan(perm_beta) else None, None, df_perm.shape[0])
            except Exception:
                pass

    perm_coefs = np.array([x for x in perm_coefs if not np.isnan(x)])
    p_value = float(np.mean(np.abs(perm_coefs) >= np.abs(beta_obs))) if perm_coefs.size > 0 and not np.isnan(beta_obs) else None

    return float(beta_obs) if not np.isnan(beta_obs) else None, p_value, perm_coefs


# ------------ PROCESSA UN GENE (NEI PROCESSI PARALLELI) ------------
def process_single_gene(gene_col, gene_original, Ecols):

    print(f"[START] {gene_original}")

    # ogni processo legge il DF dal pickle
    df = pickle.load(open(TEMP_DF_PATH, "rb"))

    if gene_already_done(gene_original):
        print(f"[SKIP] {gene_original}")
        return None

    # check counts
    treated_df = df[df[gene_col] == 1]
    control_df = df[df[gene_col] == 0]
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

    # ---------- PERMUTATION TEST ON MATCHED SAMPLE ----------
    seed = random_state + (abs(hash(gene_col)) % 2_000_000)
    try:
        beta_obs, p_emp, perm_coefs = permutation_test_interaction(
            matched,
            formula,
            gene_col,
            interaction_name,
            n_perm=n_perm,
            save_iter=False,
            gene_original=gene_original,
            seed=seed,
        )
    except Exception as e:
        print(f"[PERM ERROR] {gene_original}: {e}")
        beta_obs, p_emp, perm_coefs = obs_coef, None, np.array([])

    # ---------- SAVE RESULTS ----------
    try:
        save_gene_result(
            gene_original,
            int(matched[gene_col].sum()),  # treated in matched
            int((matched[gene_col] == 0).sum()),  # controls matched
            obs_coef,
            float(np.mean(perm_coefs)) if perm_coefs.size > 0 else None,
            float(np.std(perm_coefs)) if perm_coefs.size > 0 else None,
            p_emp,
            None,
        )
    except Exception as e:
        print(f"[SAVE ERROR] {gene_original}: {e}")

    print(f"[DONE] {gene_original}")
    return gene_original


# ------------ MAIN SCRIPT ---------------
def main():

    # --------- LOAD GENETIC ---------
    df_gen = pd.read_csv(raw_file, sep=sep, decimal=decimal)
    non_gen_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]
    # drop columns with too many missing "-1"
    df_gen = df_gen.loc[:, (df_gen == -1).mean() < 0.30]
    gene_cols = [c for c in df_gen.columns if c not in non_gen_cols]

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

    print(f"Totale geni: {len(gene_cols_safe)}")

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


if __name__ == "__main__":
    main()
