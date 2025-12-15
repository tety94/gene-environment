import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.multitest import multipletests

def add_fdr(df, p_col="empirical_p", fdr_col="fdr"):
    df = df.copy()
    pvals = df[p_col].astype(float).values
    df[fdr_col] = multipletests(pvals, method="fdr_bh")[1]
    return df

def volcano_plot(df, beta_col="obs_coef", p_col="empirical_p", gene_col="gene",
                 p_thresh=1e-5, fdr_col="fdr", fdr_thresh=0.05, save_path=None):
    df = df.copy()
    df["neglog10p"] = -np.log10(df[p_col])
    plt.figure(figsize=(9, 7))
    plt.scatter(df[beta_col], df["neglog10p"], alpha=0.6)
    plt.axhline(-np.log10(p_thresh), linestyle="--", color="red", label=f"p = {p_thresh}")
    if fdr_col in df.columns:
        sig_fdr = df[df[fdr_col] < fdr_thresh]
        plt.scatter(sig_fdr[beta_col], sig_fdr["neglog10p"],
                    s=50, edgecolor="black", label=f"FDR < {fdr_thresh}", color="orange")
    plt.xlabel("Beta dell'interazione")
    plt.ylabel("-log10(p)")
    plt.title("Volcano Plot: Interazioni Gene × Ambiente")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
