import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from concurrent.futures import ProcessPoolExecutor, as_completed
from db import gene_already_done, save_gene_iteration, save_gene_result

# ---------------- CONFIG ----------------
raw_file = "gen_diminuito.csv"
env_file = "componenti_ambientali.csv"
sep = ';'
decimal = '.'
onset_col = "onset_age"
exposures = ["seminativi_1000", "vigneti_1000"]
covariates = ["sex"]
n_iter = 250
random_state = 42
standardize = True
min_treated = 5
min_sample_size = 10
max_workers = 6   # CPU parallele
# ---------------------------------------

np.random.seed(random_state)
random.seed(random_state)


def build_formula(onset_col, gene_col, exposures, covariates, df_subset):
    exposures_str = " + ".join(exposures)
    formula = f"{onset_col} ~ {gene_col} * ({exposures_str})"
    cov_in_df = [c for c in covariates if c in df_subset.columns]
    if cov_in_df:
        formula += " + " + " + ".join(cov_in_df)
    return formula


# ------------ PROCESSA UN GENE (USATO NEI PROCESSI PARALLELI) ------------
def process_single_gene(gene_col, gene_original, df, Ecols):

    print(f"[START] {gene_original}")

    if gene_already_done(gene_original):
        print(f"[SKIP] {gene_original}")
        return None

    treated_df = df[df[gene_col] == 1].reset_index(drop=True)
    control_df = df[df[gene_col] == 0].reset_index(drop=True)
    n_treated = len(treated_df)

    if n_treated < min_treated or len(control_df) == 0:
        print(f"[SKIP] Too few treated or controls for {gene_original}")
        return None

    # -------- COEFFICIENTE OSSERVATO --------
    obs_coef = np.nan
    try:
        cols = [onset_col, gene_col] + Ecols + covariates
        df_model = df[cols].dropna()
        if df_model.shape[0] >= min_sample_size:
            formula = build_formula(onset_col, gene_col, Ecols, covariates, df_model)
            mod = smf.ols(formula=formula, data=df_model).fit()
            inter = [n for n in mod.params.index if gene_col in n and ":" in n]
            if inter:
                obs_coef = mod.params[inter[0]]
    except Exception as e:
        print(f"[ERROR obs] {gene_original}: {e}")

    # -------- MONTE CARLO --------
    coefs = []
    for it in range(n_iter):
        try:
            sampled_ctrl = control_df.sample(n=n_treated, replace=True, random_state=random_state + it)
            sampled = pd.concat([treated_df, sampled_ctrl])

            cols = [onset_col, gene_col] + Ecols + covariates
            df_ols = sampled[cols].dropna()

            coef_val = np.nan
            p_val = np.nan

            if df_ols.shape[0] >= min_sample_size:
                formula = build_formula(onset_col, gene_col, Ecols, covariates, df_ols)
                mod = smf.ols(formula=formula, data=df_ols).fit()
                inter = [n for n in mod.params.index if gene_col in n and ":" in n]
                if inter:
                    coef_val = float(mod.params[inter[0]])
                    p_val = float(mod.pvalues.get(inter[0], np.nan))

            coefs.append(coef_val)

            # salva subito nel DB
            save_gene_iteration(gene_original, it, coef_val, p_val, df_ols.shape[0])

        except Exception as e:
            print(f"[ITER ERROR] {gene_original} iter {it}: {e}")

    # ------- STATISTICHE FINALI --------
    valid = [c for c in coefs if not np.isnan(c)]
    emp_p = np.mean(np.abs(valid) >= np.abs(obs_coef)) if valid and not np.isnan(obs_coef) else None

    save_gene_result(
        gene_original,
        n_treated,
        len(control_df),
        obs_coef,
        np.mean(valid) if valid else None,
        np.std(valid) if valid else None,
        emp_p
    )

    print(f"[DONE] {gene_original}")
    return gene_original


# ------------ MAIN SCRIPT ---------------
def main():

    # --------- LOAD GENETIC ---------
    df_gen = pd.read_csv(raw_file, sep=sep, decimal=decimal)
    non_gen_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]
    df_gen = df_gen.loc[:, (df_gen == -1).mean() < 0.30]
    gene_cols = [c for c in df_gen.columns if c not in non_gen_cols]

    for g in gene_cols:
        df_gen[g] = (df_gen[g] > 0).astype(int)

    if "IID" in df_gen.columns:
        df_gen = df_gen.rename(columns={"IID": "id"})

    # --------- LOAD ENV ---------
    df_env = pd.read_csv(env_file, sep=sep, decimal=decimal)
    df_env['id'] = df_env.get('dna').combine_first(df_env.get('codice_genome'))
    df = pd.merge(df_env, df_gen, on="id", how="inner")
    df[onset_col] = pd.to_numeric(df[onset_col], errors="coerce")

    # --------- STANDARDIZE ---------
    Ecols = []
    for exp in exposures:
        df[exp] = pd.to_numeric(df[exp], errors='coerce')
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

    # --------- PARALLEL PROCESSING ---------
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_single_gene, gc, mapping[gc], df, Ecols)
            for gc in gene_cols_safe
        ]

        for f in as_completed(futures):
            _ = f.result()  # solo per sollevare eccezioni eventuali


if __name__ == "__main__":
    main()
