import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import RAW_FILE, ENV_FILE, SEP, DECIMAL, EXPOSURES, STANDARDIZE, ONSET_COL

def load_and_prepare_data():
    # ---------- LOAD GENETIC ----------
    df_gen = pd.read_csv(RAW_FILE, sep=SEP, decimal=DECIMAL)
    non_gen_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]
    df_gen = df_gen.loc[:, (df_gen == -1).mean() < 0.30]
    gene_cols = [c for c in df_gen.columns if c not in non_gen_cols]

    for g in gene_cols:
        df_gen[g] = (df_gen[g] > 0).astype(int)

    if "IID" in df_gen.columns:
        df_gen = df_gen.rename(columns={"IID": "id"})

    # ---------- LOAD ENV ----------
    df_env = pd.read_csv(ENV_FILE, sep=SEP, decimal=DECIMAL)
    df_env["sex"] = df_env["sex"].astype("category")
    df_env["onset_site"] = df_env["onset_site"].astype("category")

    df = pd.merge(df_env, df_gen, on="id", how="inner")
    df[ONSET_COL] = pd.to_numeric(df[ONSET_COL], errors="coerce")

    # ---------- STANDARDIZE ----------
    Ecols = []
    for exp in EXPOSURES:
        df[exp] = pd.to_numeric(df[exp], errors="coerce")
        if STANDARDIZE:
            df[exp + "_std"] = StandardScaler().fit_transform(df[[exp]])
            Ecols.append(exp + "_std")
        else:
            Ecols.append(exp)

    # ---------- SAFE GENE NAMES ----------
    safe = {g: f"gene_{i}" for i, g in enumerate(gene_cols)}
    df.rename(columns=safe, inplace=True)
    gene_cols_safe = list(safe.values())
    mapping = {v: k for k, v in safe.items()}

    return df, gene_cols_safe, mapping, Ecols
