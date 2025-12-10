#test fallito
# praticamente cercavo di vedere se direttamente senza modificare i dati si vedesse una relazione gene ambiente per i PON


# montecarlo_onset_ols.py
# pip install pandas numpy statsmodels scikit-learn matplotlib

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
# File input
fn_env = "componenti_ambientali.csv"
sep = ';'
decimal = '.'

# Variabili
gene_col = "is_C9"  # colonna 0/1 che indica mutato
exposures = ["seminativi_1000"]  # lista esposizioni
covariates = ["sex"]  # eventuali covariate
onset_col = "onset_age"  # outcome

# Monte Carlo
n_iter = 1000
random_state = 42
alpha = 0.05
standardize = True

# Output
output_prefix = "montecarlo_ols"

# ---------------------------------------

np.random.seed(random_state)
random.seed(random_state)

# load dataframe
df = pd.read_csv(fn_env, sep=sep, decimal=decimal)

# assicuriamoci che gene e onset siano numerici
df[gene_col] = pd.to_numeric(df[gene_col]).fillna(0).astype(int)
df[onset_col] = pd.to_numeric(df[onset_col], errors='coerce')

# standardizza esposizioni
Ecols = []
for exp in exposures:
    df[exp] = pd.to_numeric(df[exp], errors='coerce')
    if standardize:
        scaler = StandardScaler()
        df[exp + "_std"] = scaler.fit_transform(df[[exp]])
        Ecols.append(exp + "_std")
    else:
        Ecols.append(exp)

# separa mutati e controlli
treated_df = df[df[gene_col] == 1].reset_index(drop=True)
control_df = df[df[gene_col] == 0].reset_index(drop=True)
n_treated = len(treated_df)
print(f"Mutati: {n_treated}, Non-mutati disponibili: {len(control_df)}")

# coefficiente osservato full sample
obs_coef = np.nan
try:
    cols_model = [onset_col, gene_col] + Ecols + [c for c in covariates if c in df.columns]
    df_model = df[cols_model].dropna()
    if df_model.shape[0] >= 10:
        exposures_str = " + ".join(Ecols)
        formula = f"{onset_col} ~ {gene_col} * ({exposures_str})"
        if covariates:
            cov_str = " + ".join([c for c in covariates if c in df_model.columns])
            formula += " + " + cov_str
        mod = smf.ols(formula=formula, data=df_model).fit()
        inter_name = [n for n in mod.params.index if gene_col in n and ':' in n]
        if inter_name:
            obs_coef = mod.params[inter_name[0]]
except Exception as e:
    print("Errore calcolo coefficiente osservato:", e)

# Monte Carlo storage
ols_stats = []

# Monte Carlo sampling
for it in range(n_iter):
    sampled_ctrl = control_df.sample(n=n_treated, replace=False, random_state=random_state + it)
    sampled = pd.concat([treated_df, sampled_ctrl], ignore_index=True)

    ols_res = {"iter": it, "ols_coef": np.nan, "n": 0}
    try:
        cols_model = [onset_col, gene_col] + Ecols + [c for c in covariates if c in sampled.columns]
        df_ols = sampled[cols_model].dropna()
        if df_ols.shape[0] >= 10:
            exposures_str = " + ".join(Ecols)
            formula = f"{onset_col} ~ {gene_col} * ({exposures_str})"
            if covariates:
                cov_str = " + ".join([c for c in covariates if c in df_ols.columns])
                formula += " + " + cov_str
            mod = smf.ols(formula=formula, data=df_ols).fit()
            inter_name = [n for n in mod.params.index if gene_col in n and ':' in n]
            if inter_name:
                ols_res["ols_coef"] = mod.params[inter_name[0]]
            ols_res["n"] = df_ols.shape[0]
    except:
        pass
    ols_stats.append(ols_res)

# converti in DataFrame e calcola p-value empirico Monte Carlo
ols_df = pd.DataFrame(ols_stats)
ols_df_nonnull = ols_df.dropna(subset=["ols_coef"])
mc_pval = np.mean(np.sign(ols_df_nonnull["ols_coef"]) != np.sign(obs_coef)) * 2  # two-sided

print("Monte Carlo iterations with valid OLS:", ols_df_nonnull.shape[0])
print(f"OLS interaction coef (observed): {obs_coef:.4f}")
print(f"OLS interaction coef: mean {ols_df_nonnull['ols_coef'].mean():.4f}, sd {ols_df_nonnull['ols_coef'].std():.4f}")
print(f"Monte Carlo empirical p-value: {mc_pval:.4f}")

# salva risultati
ols_df.to_csv(output_prefix + "_ols_distribution.csv", index=False, sep=';', decimal=',')

# histogrammi
plt.figure(figsize=(8, 4))
plt.hist(ols_df_nonnull["ols_coef"], bins=50, color='skyblue', edgecolor='k')
plt.axvline(obs_coef, color='red', linestyle='--', label='Obs coef full sample')
plt.title("Distribution of OLS interaction coef")
plt.legend()
plt.tight_layout()
plt.savefig(output_prefix + "_ols_hist.png", dpi=150)
plt.show()

print("Files saved with prefix:", output_prefix)
