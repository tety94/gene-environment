import pandas as pd
import numpy as np
from datetime import datetime
from config import OUTPUT_FOLDER

# ---------------- CONFIG ----------------
RAW_FILE = OUTPUT_FOLDER + "/full_chr.csv"
OUT_PARQUET = OUTPUT_FOLDER +"/gen.parquet"
SEP = ","
DECIMAL = "."

non_gen_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]

# ---------------- LOAD ----------------
print(f"[START] Carico CSV genetico: {datetime.now()}")
df = pd.read_csv(
    RAW_FILE,
    sep=SEP,
    decimal=DECIMAL,
    low_memory=False
)

# ---------------- COLONNE VARIANTI ----------------
variant_cols = [c for c in df.columns if c not in non_gen_cols]
print(f"[INFO] Varianti trovate: {len(variant_cols)}")

# ---------------- BINARIZZAZIONE VELOCE ----------------
print(f"[START] Binarizzazione genotipi: {datetime.now()}")

arr = df[variant_cols].to_numpy(dtype=np.int8, copy=False)
arr[arr > 0] = 1     # 1/2 → 1
# -1 e 0 restano invariati
df[variant_cols] = arr

# ---------------- RENAME ID ----------------
if "IID" in df.columns and "id" not in df.columns:
    df = df.rename(columns={"IID": "id"})

# ---------------- SAVE PARQUET ----------------
print(f"[START] Salvataggio Parquet: {datetime.now()}")
df.to_parquet(
    OUT_PARQUET,
    engine="pyarrow",
    compression="zstd"
)

print(f"[DONE] File pronto: {OUT_PARQUET}")
print("👉 Da ora in poi carica SOLO il parquet (non rifare mai più questa operazione)")
