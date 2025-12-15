import pandas as pd
import numpy as np
from datetime import datetime
from config import OUTPUT_FOLDER

# ---------------- CONFIG ----------------
RAW_FILE = f"{OUTPUT_FOLDER}/full_chr.csv"
OUT_PARQUET = f"{OUTPUT_FOLDER}/gen.parquet"
SEP = ","
DECIMAL = "."
NON_GEN_COLS = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]

# ---------------- START ----------------
print(f"[START] Carico CSV genetico: {datetime.now()}")
df = pd.read_csv(
    RAW_FILE,
    sep=SEP,
    decimal=DECIMAL,
    low_memory=False
)

# ---------------- VARIANT COLS ----------------
variant_cols = [c for c in df.columns if c not in NON_GEN_COLS]
print(f"[INFO] Varianti trovate: {len(variant_cols)}")

# ---------------- BINARIZZAZIONE ----------------
print(f"[START] Binarizzazione genotipi: {datetime.now()}")
arr = df[variant_cols].to_numpy(dtype=np.int8, copy=False)
arr[arr < 0] = 0  # -1 -> 0 (missing come 0)
arr[arr > 0] = 1  # 1/2 -> 1
df[variant_cols] = arr

# ---------------- RENAME ID ----------------
if "IID" in df.columns and "id" not in df.columns:
    df = df.rename(columns={"IID": "id"})

# ---------------- SAVE PARQUET ----------------
print(f"[START] Salvataggio Parquet: {datetime.now()}")
df.to_parquet(
    OUT_PARQUET,
    engine="pyarrow",
    compression="zstd",
    index=False
)

print(f"[DONE] File pronto: {OUT_PARQUET}")
print("👉 Da ora in poi carica SOLO il parquet (non rifare mai più questa operazione)")
