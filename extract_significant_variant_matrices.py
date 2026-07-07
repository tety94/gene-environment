#!/usr/bin/env python3
"""
Estrae, per le varianti significative in ENTRAMBE le coorti (generation 1 e 2,
secondo db.get_significant_results()), il genotipo binario (0 = non mutato,
1 = mutato) per ciascun paziente, producendo due CSV (uno per coorte)
pazienti x varianti.

Fonte varianti significative : db.get_significant_results()
Fonte genotipi               : i file genetici RAW per generazione (stesso formato
                                usato da data_loader.py: colonna id/IID + una colonna
                                per variante, nome "{chrom}_{pos}_{mutation}",
                                valori dosaggio 0/1/2/-1/. ecc.)

Usage: python extract_significant_variant_matrices.py
"""
import os
import pandas as pd

from db import get_significant_results

# ── CONFIG ────────────────────────────────────────────────────────────────
GENOTYPE_FILES = {
    1: "/mnt/cresla_prod/genome_datasets/merged_csv/full_chr_gen1_test1.csv",
    2: "/mnt/cresla_prod/genome_datasets/merged_csv/gen2_variants.csv",
}
OUT_DIR = "/srv/python-projects/gene-environment/significant_variant_matrices"
# ─────────────────────────────────────────────────────────────────────────────


def build_variant_label(chromosome, position, mutation):
    return f"{chromosome}_{position}_{mutation}"


def load_id_col(header):
    """Trova la colonna id, gestendo il rename IID -> id fatto da data_loader.py."""
    if "id" in header:
        return "id"
    if "IID" in header:
        return "IID"
    raise ValueError(f"Nessuna colonna id/IID trovata nell'header: {header[:10]}...")


def binarize(df_geno):
    """
    Converte dosaggio in presenza mutazione binaria:
      0            -> 0 (non mutato)
      1 oppure 2   -> 1 (mutato)
      qualsiasi altro valore (-1, ., NaN, ...) -> NaN (missing, escluso a valle)
    """
    def conv(x):
        s = str(x).strip()
        if s == "0":
            return 0
        if s in ("1", "2"):
            return 1
        return pd.NA

    return df_geno.applymap(conv)


def extract_cohort(generation, variant_labels, out_dir):
    path = GENOTYPE_FILES[generation]
    if not os.path.exists(path):
        print(f"[WARN] File non trovato per generation {generation}: {path}")
        return

    header = pd.read_csv(path, nrows=0).columns.tolist()
    id_col = load_id_col(header)

    present = [v for v in variant_labels if v in header]
    missing = [v for v in variant_labels if v not in header]
    if missing:
        print(f"  [WARN] Coorte {generation}: {len(missing)} varianti non trovate nel file:")
        for m in missing:
            print(f"     - {m}")

    if not present:
        print(f"  [ERROR] Coorte {generation}: nessuna variante significativa presente nel file.")
        return

    df = pd.read_csv(path, usecols=[id_col] + present)
    if id_col != "id":
        df = df.rename(columns={id_col: "id"})
    df = df.set_index("id")

    df_bin = binarize(df)

    out_path = os.path.join(out_dir, f"cohort{generation}_significant_variants.csv")
    df_bin.to_csv(out_path)
    print(f"  -> Coorte {generation}: {out_path}  ({len(df_bin)} pazienti, {df_bin.shape[1]} varianti)")


def main():
    print("=== Recupero varianti significative (entrambe le coorti) dal DB ===")
    sig = get_significant_results()
    if sig.empty:
        print("[INFO] Nessuna variante significativa trovata in entrambe le coorti. Esco.")
        return

    sig["label"] = sig.apply(
        lambda r: build_variant_label(r["chromosome"], r["position"], r["mutation"]), axis=1
    )
    unique_labels = sorted(sig["label"].unique())
    print(f"  {len(unique_labels)} varianti uniche da estrarre.")

    os.makedirs(OUT_DIR, exist_ok=True)

    for generation in (1, 2):
        extract_cohort(generation, unique_labels, OUT_DIR)

    print("\n=== Fatto ===")


if __name__ == "__main__":
    main()