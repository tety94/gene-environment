#!/usr/bin/env python3
"""
Analizza la differenza di età d'esordio (onset_age) fra pazienti mutati e non
mutati, per ciascuna variante significativa e per ciascuna coorte (1 e 2),
a partire dalle matrici prodotte da extract_significant_variant_matrices.py.

Test statistico: Mann-Whitney U (default, non parametrico — robusto rispetto a
outlier/non normalità tipica dell'età d'esordio). Alternativa: t-test di Welch
(USE_MANN_WHITNEY = False).

Output in OUT_DIR:
  - results_table.csv         : una riga per variante x coorte (n, mediane, p-value, FDR)
  - boxplots/<variante>_cohortN.png : boxplot onset_age mutati vs non mutati
  - summary_significant.csv   : solo risultati con p_adj_fdr < ALPHA

Usage: python analyze_variant_onset_age.py
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TARGET_COL, COVARIATES

# ── CONFIG ────────────────────────────────────────────────────────────────
ONSET_FILE       = "componenti_ambientali.csv"   # contiene ENTRAMBE le generazioni
SAMPLE_ID_COL    = "id"
MATRIX_DIR       = "/srv/python-projects/gene-environment/significant_variant_matrices"
OUT_DIR          = "/srv/python-projects/gene-environment/onset_age_analysis"
USE_MANN_WHITNEY = True
ALPHA            = 0.05
MIN_GROUP_SIZE   = 5   # minimo pazienti per gruppo per eseguire il test
# ─────────────────────────────────────────────────────────────────────────────


def load_onset_age():
    df = pd.read_csv(ONSET_FILE, usecols=[SAMPLE_ID_COL, TARGET_COL] + COVARIATES)
    return df.set_index(SAMPLE_ID_COL)


def run_test(mutati, non_mutati):
    if USE_MANN_WHITNEY:
        stat, p = mannwhitneyu(mutati, non_mutati, alternative="two-sided")
    else:
        stat, p = ttest_ind(mutati, non_mutati, equal_var=False, nan_policy="omit")
    return stat, p


def make_boxplot(mutati, non_mutati, variant, cohort, out_dir):
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot([non_mutati, mutati], labels=["WT", "Mutato"], showmeans=True)
    ax.set_ylabel("Età d'esordio")
    ax.set_title(f"{variant}\ncoorte {cohort}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{variant}_cohort{cohort}.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    box_dir = os.path.join(OUT_DIR, "boxplots")
    os.makedirs(box_dir, exist_ok=True)

    df_onset = load_onset_age()

    rows = []
    for cohort in (1, 2):
        matrix_path = os.path.join(MATRIX_DIR, f"cohort{cohort}_significant_variants.csv")
        if not os.path.exists(matrix_path):
            print(f"[WARN] Manca {matrix_path}, salto coorte {cohort}")
            continue

        df_geno = pd.read_csv(matrix_path, index_col="id")
        df = df_geno.join(df_onset, how="inner")

        variants = list(df_geno.columns)
        print(f"[INFO] Coorte {cohort}: {len(df)} pazienti, {len(variants)} varianti")

        for variant in variants:
            sub = df[[variant, TARGET_COL]].dropna()
            sub[variant] = pd.to_numeric(sub[variant], errors="coerce")
            mutati = sub.loc[sub[variant] == 1, TARGET_COL]
            non_mutati = sub.loc[sub[variant] == 0, TARGET_COL]

            if len(mutati) < MIN_GROUP_SIZE or len(non_mutati) < MIN_GROUP_SIZE:
                print(f"  [SKIP] {variant} coorte {cohort}: gruppi troppo piccoli "
                      f"(mutati={len(mutati)}, non_mutati={len(non_mutati)})")
                continue

            stat, p = run_test(mutati, non_mutati)
            make_boxplot(mutati, non_mutati, variant, cohort, box_dir)

            rows.append({
                "variant": variant,
                "cohort": cohort,
                "n_mutati": len(mutati),
                "n_non_mutati": len(non_mutati),
                "median_onset_mutati": mutati.median(),
                "median_onset_non_mutati": non_mutati.median(),
                "delta_median": mutati.median() - non_mutati.median(),
                "statistic": stat,
                "p_value": p,
            })

    if not rows:
        print("[INFO] Nessun risultato calcolato. Esco.")
        return

    results = pd.DataFrame(rows)
    results["p_adj_fdr"] = np.nan
    for cohort in results["cohort"].unique():
        mask = results["cohort"] == cohort
        _, p_adj, _, _ = multipletests(results.loc[mask, "p_value"], method="fdr_bh")
        results.loc[mask, "p_adj_fdr"] = p_adj

    results = results.sort_values(["cohort", "p_value"])
    results.to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)

    significant = results[results["p_adj_fdr"] < ALPHA]
    significant.to_csv(os.path.join(OUT_DIR, "summary_significant.csv"), index=False)

    print("\n=== Fatto ===")
    print(f"  Tabella completa : {OUT_DIR}/results_table.csv")
    print(f"  Significativi FDR: {OUT_DIR}/summary_significant.csv  ({len(significant)} righe)")
    print(f"  Boxplot          : {box_dir}/")


if __name__ == "__main__":
    main()