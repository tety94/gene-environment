import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from config import RAW_FILE, ENV_FILE, SEP, DECIMAL, EXPOSURES, STANDARDIZE, ONSET_COL

def load_and_prepare_data():
    # ---------- LOAD GENETIC ----------
    print(f"[START] Carico file genetica: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df_gen = pd.read_csv(RAW_FILE, sep=SEP, decimal=DECIMAL, low_memory=False)
    print(f"[START] Trovo colonne: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    non_gen_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]
    variant_cols = [c for c in df_gen.columns if c not in non_gen_cols]

    print(f"[START] metto type int: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    arr = df_gen[variant_cols].to_numpy(dtype=np.int8, copy=False)
    arr[arr < 0] = 0  # -1 -> 0 (missing come 0)
    arr[arr > 0] = 1  # 1/2 -> 1
    df_gen[variant_cols] = arr

    if "IID" in df_gen.columns:
        df_gen = df_gen.rename(columns={"IID": "id"})

    # ---------- LOAD ENV ----------
    print(f"[START] Carico file ambientali: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df_env = pd.read_csv(ENV_FILE, sep=SEP, decimal=DECIMAL)
    df_env["sex"] = df_env["sex"].astype("category")
    df_env["onset_site"] = df_env["onset_site"].astype("category")

    print(f"[START] Merge file gene - ambiente: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df = pd.merge(df_env, df_gen, on="id", how="inner")
    df[ONSET_COL] = pd.to_numeric(df[ONSET_COL], errors="coerce")

    # ---------- STANDARDIZE ----------
    print(f"[START] Inizio standardizzazione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    Ecols = []
    for exp in EXPOSURES:
        df[exp] = pd.to_numeric(df[exp], errors="coerce")
        if STANDARDIZE:
            df[exp + "_std"] = StandardScaler().fit_transform(df[[exp]])
            Ecols.append(exp + "_std")
        else:
            Ecols.append(exp)

    # ---------- SAFE GENE NAMES ----------
    print(f"[START] Creo dizionario per salvare i nomi delle variabili: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe = {g: f"variant_{i}" for i, g in enumerate(variant_cols)}
    df.rename(columns=safe, inplace=True)
    variant_cols_safe = list(safe.values())
    mapping = {v: k for k, v in safe.items()}

    return df, variant_cols_safe, mapping, Ecols
