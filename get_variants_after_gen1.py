#!/usr/bin/env python3
"""
Estrae varianti specifiche da file VCF.gz divisi per cromosoma e generazione.
Produce due CSV (uno per generazione) con pazienti in righe e varianti in colonne,
poi li concatena in un unico CSV finale.

Usage: python extract_variants.py
"""

import os
import gzip
import csv
import pandas as pd
from collections import defaultdict

component = 'vigneti'
# ─── CONFIG ───────────────────────────────────────────────────────────────────
VARIANTS_CSV   = f"/srv/python-projects/gene-environment/variant_to_extract_{component}.csv"
GEN2_DIR       = "/mnt/cresla_prod/genome_datasets/gen2"
GEN3_DIR       = "/mnt/cresla_prod/genome_datasets/gen3"
OUT_GEN2       = "/srv/python-projects/gene-environment/output_gen2.csv"
OUT_GEN3       = "/srv/python-projects/gene-environment/output_gen3.csv"
OUT_COMBINED   = f"/srv/python-projects/gene-environment/output_combined_{component}.csv"

GEN_PREFIX = {
    GEN2_DIR: "gen2",
    GEN3_DIR: "gen3",
}
# ──────────────────────────────────────────────────────────────────────────────


def load_variants(path):
    """Legge il CSV delle varianti e restituisce una lista di dict."""
    variants = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            variants.append({
                "chrom": str(row["chromosome"]),
                "pos":   int(row["position"]),
                "mut":   row["mutation"],          # es. A_G
                "label": f"chr{row['chromosome']}_{row['position']}_{row['mutation']}",
            })
    return variants


def parse_genotype(ref, alt_field, gt_str):
    """
    Converte un genotipo VCF (0/0, 0/1, 1/1, .) in un valore numerico:
      0 = omozigote ref
      1 = eterozigote
      2 = omozigote alt
      . = missing
    """
    if gt_str in (".", "./.", ".|."):
        return "."
    # gestisce sia / che |
    sep = "/" if "/" in gt_str else "|"
    alleles = gt_str.split(sep)
    try:
        count = sum(int(a) > 0 for a in alleles)
        return count
    except ValueError:
        return "."


def extract_from_vcf(vcf_gz_path, variants_for_chrom):
    """
    Scorre un VCF.gz e raccoglie i genotipi per le varianti richieste.
    Restituisce:
      - samples: lista ordinata degli ID campione
      - results: dict { label -> { sample_id -> genotipo } }
    """
    # indice rapido per posizione
    pos_index = defaultdict(list)
    for v in variants_for_chrom:
        pos_index[v["pos"]].append(v)

    samples = []
    results = {v["label"]: {} for v in variants_for_chrom}

    with gzip.open(vcf_gz_path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                # header: #CHROM POS ID REF ALT QUAL FILTER INFO FORMAT sample1 sample2 ...
                parts = line.strip().split("\t")
                samples = parts[9:]
                continue

            parts = line.strip().split("\t")
            pos = int(parts[1])

            if pos not in pos_index:
                continue

            ref   = parts[3]
            alt   = parts[4]   # può essere multi-allele (A,T)
            fmt   = parts[8].split(":")
            gt_idx = fmt.index("GT") if "GT" in fmt else 0

            for v in pos_index[pos]:
                # verifica che ref/alt corrispondano alla mutazione attesa
                exp_ref, exp_alt = v["mut"].split("_")
                alt_alleles = alt.split(",")
                if ref != exp_ref or exp_alt not in alt_alleles:
                    continue  # variante non corrispondente in questo locus

                label = v["label"]
                for i, sample in enumerate(samples):
                    sample_data = parts[9 + i].split(":")
                    gt_str      = sample_data[gt_idx]
                    results[label][sample] = parse_genotype(ref, alt, gt_str)

    return samples, results


def process_generation(gen_dir, variants, out_csv):
    """
    Per una generazione, itera i cromosomi necessari, estrae i genotipi
    e scrive il CSV (pazienti x varianti).
    """
    prefix = GEN_PREFIX[gen_dir]

    # raggruppa varianti per cromosoma
    by_chrom = defaultdict(list)
    for v in variants:
        by_chrom[v["chrom"]].append(v)

    # raccoglie tutti i campioni (potrebbero variare per cromosoma, ma di solito no)
    all_samples = None
    # results globali: label -> {sample -> gt}
    global_results = {v["label"]: {} for v in variants}

    for chrom, chrom_variants in sorted(by_chrom.items()):
        vcf_path = os.path.join(gen_dir, f"{prefix}_vcf_chr{chrom}.vcf.gz")
        if not os.path.exists(vcf_path):
            print(f"  [WARN] File non trovato: {vcf_path}")
            continue

        print(f"  Elaborando {vcf_path} ({len(chrom_variants)} varianti)...")
        samples, results = extract_from_vcf(vcf_path, chrom_variants)

        if all_samples is None:
            all_samples = samples
        # aggiorna results globali
        for label, sample_dict in results.items():
            global_results[label].update(sample_dict)

    if all_samples is None:
        print(f"  [ERROR] Nessun campione trovato per {gen_dir}")
        return None

    # costruisce DataFrame: righe=pazienti, colonne=varianti
    variant_labels = [v["label"] for v in variants]
    rows = []
    for sample in all_samples:
        row = {"sample_id": sample}
        for label in variant_labels:
            row[label] = global_results[label].get(sample, ".")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("sample_id")
    df.to_csv(out_csv)
    print(f"  → Salvato: {out_csv}  ({len(df)} pazienti, {len(variant_labels)} varianti)")
    return df


def main():
    print("=== Caricamento varianti ===")
    variants = load_variants(VARIANTS_CSV)
    print(f"  {len(variants)} varianti da estrarre:")
    for v in variants:
        print(f"    {v['label']}")

    print("\n=== Generazione 2 ===")
    df2 = process_generation(GEN2_DIR, variants, OUT_GEN2)

    print("\n=== Generazione 3 ===")
    df3 = process_generation(GEN3_DIR, variants, OUT_GEN3)

    # ── Concatenazione ────────────────────────────────────────────────────────
    if df2 is not None and df3 is not None:
        print("\n=== Concatenazione ===")
        # pd.concat allinea automaticamente le colonne per nome
        df_combined = pd.concat([df2, df3], axis=0, join="outer")
        # riordina le colonne nell'ordine originale delle varianti
        col_order = [v["label"] for v in variants]
        # mantieni solo le colonne presenti
        col_order = [c for c in col_order if c in df_combined.columns]
        df_combined = df_combined[col_order]
        df_combined.fillna(".", inplace=True)
        df_combined.to_csv(OUT_COMBINED)
        print(f"  → Salvato: {OUT_COMBINED}  ({len(df_combined)} pazienti totali)")

    print("\n=== Fatto ===")


if __name__ == "__main__":
    main()