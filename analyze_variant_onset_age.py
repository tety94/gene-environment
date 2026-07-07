#!/usr/bin/env python3
"""
Analizza la differenza di età d'esordio (onset_age) fra pazienti mutati e non
mutati, per ciascuna variante significativa e per ciascuna coorte (1 e 2),
a partire dalle matrici prodotte da extract_significant_variant_matrices.py.

Test statistico: Mann-Whitney U (default, non parametrico — robusto rispetto a
outlier/non normalità tipica dell'età d'esordio). Alternativa: t-test di Welch
(USE_MANN_WHITNEY = False).

Output in OUT_DIR:
  - results_table.csv              : una riga per variante x coorte (n, mediane,
                                      effect size, p-value, FDR, IC bootstrap)
  - summary_significant.csv        : solo risultati con p_adj_fdr < ALPHA
  - combined_results_table.csv     : una riga per variante, coorte 1 e coorte 2
                                      affiancate (formato tipo supplementary table)
  - boxplots/<variante>_cohortN.png       : boxplot onset_age mutati vs non mutati, singola coorte
  - boxplots_combined/<variante>.png      : stessa variante, coorte 1 e 2 affiancate nello stesso pannello
  - forest_plot.png                       : riepilogo di tutte le varianti/coorti,
                                             delta mediana con IC 95% bootstrap

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
ONSET_FILE          = "data/onset_age.csv"   # contiene ENTRAMBE le generazioni
SAMPLE_ID_COL        = "id"
MATRIX_DIR           = "/srv/python-projects/gene-environment/significant_variant_matrices"
OUT_DIR              = "/srv/python-projects/gene-environment/onset_age_analysis"
USE_MANN_WHITNEY     = True
ALPHA                = 0.05
MIN_GROUP_SIZE       = 5    # minimo pazienti per gruppo per eseguire il test
LOW_POWER_THRESHOLD  = 10   # sotto questa soglia per gruppo, flag low_power=True in tabella
N_BOOT               = 2000  # numero resample bootstrap per IC del delta mediana
RANDOM_STATE         = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_onset_age():
    df = pd.read_csv(ONSET_FILE, usecols=[SAMPLE_ID_COL, TARGET_COL] + COVARIATES)
    return df.set_index(SAMPLE_ID_COL)


def run_test(mutati, non_mutati):
    n1, n2 = len(mutati), len(non_mutati)
    if USE_MANN_WHITNEY:
        stat, p = mannwhitneyu(mutati, non_mutati, alternative="two-sided")
        # rank-biserial correlation: 1 - 2U/(n1*n2). U qui è quella riferita a 'mutati'.
        effect_size = 1 - (2 * stat) / (n1 * n2)
    else:
        stat, p = ttest_ind(mutati, non_mutati, equal_var=False, nan_policy="omit")
        pooled_sd = np.sqrt(((n1 - 1) * mutati.std(ddof=1) ** 2 + (n2 - 1) * non_mutati.std(ddof=1) ** 2) / (n1 + n2 - 2))
        effect_size = (mutati.mean() - non_mutati.mean()) / pooled_sd  # Cohen's d
    return stat, p, effect_size


def bootstrap_median_diff_ci(mutati, non_mutati, n_boot=N_BOOT, alpha=ALPHA, seed=RANDOM_STATE):
    """IC bootstrap (percentile) per il delta di mediana (mutati - non_mutati)."""
    rng = np.random.default_rng(seed)
    mutati_arr = np.asarray(mutati)
    non_mutati_arr = np.asarray(non_mutati)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        m_sample = rng.choice(mutati_arr, size=len(mutati_arr), replace=True)
        nm_sample = rng.choice(non_mutati_arr, size=len(non_mutati_arr), replace=True)
        diffs[i] = np.median(m_sample) - np.median(nm_sample)

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def make_boxplot(mutati, non_mutati, variant, cohort, out_dir):
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot([non_mutati, mutati], labels=["WT", "Mutato"], showmeans=True)
    ax.set_ylabel("Età d'esordio")
    ax.set_title(f"{variant}\ncoorte {cohort}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{variant}_cohort{cohort}.png"), dpi=150)
    plt.close(fig)


def make_combined_boxplot(groups_for_variant, variant, out_dir):
    """Boxplot con un pannello per coorte, stessa scala y, per confronto diretto."""
    cohorts_present = sorted(groups_for_variant.keys())
    fig, axes = plt.subplots(1, len(cohorts_present), figsize=(4 * len(cohorts_present), 5), sharey=True)
    if len(cohorts_present) == 1:
        axes = [axes]

    for ax, cohort in zip(axes, cohorts_present):
        mutati, non_mutati = groups_for_variant[cohort]
        ax.boxplot([non_mutati, mutati], labels=["WT", "Mutato"], showmeans=True)
        ax.set_title(f"Coorte {cohort}")

    axes[0].set_ylabel("Età d'esordio")
    fig.suptitle(variant)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{variant}_combined.png"), dpi=150)
    plt.close(fig)


def make_combined_table(results):
    """Tabella wide: una riga per variante, colonne coorte 1 e coorte 2 affiancate."""
    wide = results.set_index(["variant", "cohort"]).unstack("cohort")
    wide.columns = [f"{col}_c{cohort}" for col, cohort in wide.columns]
    return wide.reset_index()


def make_forest_plot(results, out_path):
    """Riepilogo unico: delta mediana (mutati - non mutati) con IC 95% bootstrap,
    una riga per variante x coorte, colorata per coorte."""
    results_sorted = results.sort_values(["variant", "cohort"]).reset_index(drop=True)
    if results_sorted.empty:
        return

    y_labels = [f"{r.variant}  (coorte {r.cohort})" for r in results_sorted.itertuples()]
    y_pos = np.arange(len(results_sorted))[::-1]  # dall'alto in basso

    xerr_low = results_sorted["delta_median"] - results_sorted["ci_low"]
    xerr_high = results_sorted["ci_high"] - results_sorted["delta_median"]
    colors = results_sorted["cohort"].map({1: "tab:blue", 2: "tab:orange"}).fillna("gray")

    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(results_sorted))))
    ax.errorbar(
        results_sorted["delta_median"], y_pos,
        xerr=[xerr_low, xerr_high],
        fmt="none", ecolor="gray", capsize=3, zorder=1,
    )
    ax.scatter(results_sorted["delta_median"], y_pos, c=colors, zorder=2)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Δ mediana onset_age (mutati - non mutati)  [IC 95% bootstrap]")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", label="Coorte 1"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", label="Coorte 2"),
    ]
    ax.legend(handles=handles, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    box_dir = os.path.join(OUT_DIR, "boxplots")
    box_combined_dir = os.path.join(OUT_DIR, "boxplots_combined")
    os.makedirs(box_dir, exist_ok=True)
    os.makedirs(box_combined_dir, exist_ok=True)

    df_onset = load_onset_age()

    rows = []
    groups = {}  # groups[variant][cohort] = (mutati_series, non_mutati_series)

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

            stat, p, effect_size = run_test(mutati, non_mutati)
            ci_low, ci_high = bootstrap_median_diff_ci(mutati, non_mutati)
            make_boxplot(mutati, non_mutati, variant, cohort, box_dir)

            groups.setdefault(variant, {})[cohort] = (mutati, non_mutati)

            rows.append({
                "variant": variant,
                "cohort": cohort,
                "n_mutati": len(mutati),
                "n_non_mutati": len(non_mutati),
                "median_onset_mutati": mutati.median(),
                "median_onset_non_mutati": non_mutati.median(),
                "delta_median": mutati.median() - non_mutati.median(),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "statistic": stat,
                "effect_size": effect_size,  # rank-biserial (MW) o Cohen's d (t-test)
                "p_value": p,
                "low_power": (len(mutati) < LOW_POWER_THRESHOLD) or (len(non_mutati) < LOW_POWER_THRESHOLD),
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

    # ── tabella combinata coorte 1 / coorte 2 affiancate ─────────────────────
    combined_table = make_combined_table(results)
    combined_table.to_csv(os.path.join(OUT_DIR, "combined_results_table.csv"), index=False)

    # ── boxplot combinati per variante (coorte 1 e 2 nello stesso pannello) ──
    for variant, groups_for_variant in groups.items():
        make_combined_boxplot(groups_for_variant, variant, box_combined_dir)

    # ── forest plot riepilogativo ─────────────────────────────────────────────
    make_forest_plot(results, os.path.join(OUT_DIR, "forest_plot.png"))

    print("\n=== Fatto ===")
    print(f"  Tabella per coorte    : {OUT_DIR}/results_table.csv")
    print(f"  Significativi FDR     : {OUT_DIR}/summary_significant.csv  ({len(significant)} righe)")
    print(f"  Tabella combinata     : {OUT_DIR}/combined_results_table.csv")
    print(f"  Boxplot singoli       : {box_dir}/")
    print(f"  Boxplot combinati     : {box_combined_dir}/")
    print(f"  Forest plot           : {OUT_DIR}/forest_plot.png")


if __name__ == "__main__":
    main()