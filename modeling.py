import statsmodels.formula.api as smf
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from config import TARGET_COL, MATCH_K, MIN_TREATED, MIN_SAMPLE_SIZE, N_PERM, RANDOM_STATE, MIN_OBS_COEF, N_PERM_HIGH, PVALUE_THRESHOLD
from db import save_variant_result, variant_already_done, get_conn, mark_variant_in_progress, reset_variant_in_progress
from matching import match_control_units, check_balance

global_df = None
worker_conn = None

def build_formula(onset_col, variant_col, exposures, covariates, df_subset):
    exposures_str = " + ".join(exposures)
    formula = f"{onset_col} ~ {variant_col} * ({exposures_str})"
    cov_in_df = [c for c in covariates if c in df_subset.columns]
    if cov_in_df:
        formula += " + " + " + ".join(cov_in_df)
    return formula

def _find_interaction_term(mod_params_index, variant_col):
    for name in mod_params_index:
        if ":" in name and variant_col in name:
            return name
    return None

def save_variant_result_not_calculated(conn, variant_original, muted, not_muted, max_smd =None):
    save_variant_result(conn, variant_original, muted,not_muted, None, None, None, 1, N_PERM, max_smd)
    reset_variant_in_progress(conn, variant_original, success=True)





def process_single_variant(variant_col, variant_original, Ecols):
    df = global_df  # caricato dall'initializer, non da pickle ogni volta

    # -----------------------------
    # CREA VARIABILE MATCHING BINARIA
    # -----------------------------
    df = df[df[variant_col] != '.'].copy()
    df[variant_col] = df[variant_col].astype(int)
    df["_match_variant"] = (df[variant_col] > 0).astype(int)

    n_treated = int((df["_match_variant"] == 1).sum())
    n_control = int((df["_match_variant"] == 0).sum())

    def _empty(obs_coef=None, max_smd=None, iterations=N_PERM):
        return {
            "variant": variant_original,
            "n_treated": n_treated,
            "n_control": n_control,
            "obs_coef": obs_coef,
            "perm_mean": None,
            "perm_std": None,
            "p_emp": None,
            "max_smd": max_smd,
            "iterations": iterations,
            "empirical_p_significant": False
        }

    if n_treated < MIN_TREATED or n_control == 0:
        return _empty()

    cols = [TARGET_COL, variant_col, "_match_variant"] + Ecols
    df_model = df[cols].dropna()
    if df_model.shape[0] < MIN_SAMPLE_SIZE:
        return _empty()

    matched_obs = match_control_units(df_model, "_match_variant", k=MATCH_K, covariates_for_matching=Ecols)
    if matched_obs is None or matched_obs.shape[0] < MIN_SAMPLE_SIZE:
        return _empty()

    smd_results = check_balance(matched_obs, "_match_variant", Ecols)
    max_smd = max(smd_results.values()) if smd_results else 1

    if max_smd > 0.25:
        return _empty(max_smd=max_smd)

    formula = build_formula(TARGET_COL, variant_col, Ecols, [], matched_obs)
    mod = smf.ols(formula=formula, data=matched_obs).fit()
    interaction_name = _find_interaction_term(mod.params.index, variant_col)

    if interaction_name is None:
        return None

    obs_coef = float(mod.params[interaction_name])

    n_treated_matched = int(matched_obs["_match_variant"].sum())
    n_control_matched = int((matched_obs["_match_variant"] == 0).sum())

    if abs(obs_coef) < MIN_OBS_COEF:
        return {
            "variant": variant_original,
            "n_treated": n_treated_matched,
            "n_control": n_control_matched,
            "obs_coef": obs_coef,
            "perm_mean": None,
            "perm_std": None,
            "p_emp": None,
            "max_smd": max_smd,
            "iterations": N_PERM,
            "empirical_p_significant": False
        }

    rng = np.random.RandomState(RANDOM_STATE + (abs(hash(variant_col)) % 2_000_000))

    # ======================================================
    # PERMUTAZIONI LIGHT
    # ======================================================
    perm_betas_light = []
    for _ in tqdm(range(N_PERM), desc=f"Perm LIGHT {variant_col}", leave=False):
        df_perm = df_model.copy()
        df_perm[variant_col] = rng.permutation(df_perm[variant_col].values)
        df_perm["_match_variant"] = (df_perm[variant_col] > 0).astype(int)
        matched_perm = match_control_units(df_perm, "_match_variant", k=MATCH_K, covariates_for_matching=Ecols)
        if matched_perm is None or matched_perm.shape[0] < MIN_SAMPLE_SIZE:
            perm_betas_light.append(np.nan)
            continue
        mod_perm = smf.ols(formula=formula, data=matched_perm).fit()
        perm_betas_light.append(mod_perm.params.get(interaction_name, np.nan))

    perm_betas_light = np.array([x for x in perm_betas_light if not np.isnan(x)])
    p_emp_light = float(np.mean(np.abs(perm_betas_light) >= np.abs(obs_coef))) if perm_betas_light.size > 0 else None

    # ======================================================
    # PERMUTAZIONI HIGH — solo se light è significativo
    # ======================================================
    if p_emp_light is not None and p_emp_light < PVALUE_THRESHOLD:
        n_additional = N_PERM_HIGH - N_PERM
        perm_betas_additional = []

        for _ in tqdm(range(n_additional), desc=f"Perm HIGH {variant_col}", leave=False):
            df_perm = df_model.copy()
            df_perm[variant_col] = rng.permutation(df_perm[variant_col].values)
            df_perm["_match_variant"] = (df_perm[variant_col] > 0).astype(int)
            matched_perm = match_control_units(df_perm, "_match_variant", k=MATCH_K, covariates_for_matching=Ecols)
            if matched_perm is None or matched_perm.shape[0] < MIN_SAMPLE_SIZE:
                perm_betas_additional.append(np.nan)
                continue
            mod_perm = smf.ols(formula=formula, data=matched_perm).fit()
            perm_betas_additional.append(mod_perm.params.get(interaction_name, np.nan))

        perm_betas_final = np.concatenate([perm_betas_light, perm_betas_additional])
        perm_betas_final = np.array([x for x in perm_betas_final if not np.isnan(x)])
        p_emp_final = float(np.mean(np.abs(perm_betas_final) >= np.abs(obs_coef))) if perm_betas_final.size > 0 else None
        iterations_final = N_PERM_HIGH
    else:
        perm_betas_final = perm_betas_light
        p_emp_final = p_emp_light
        iterations_final = N_PERM

    return {
        "variant": variant_original,
        "n_treated": n_treated_matched,
        "n_control": n_control_matched,
        "obs_coef": obs_coef,
        "perm_mean": float(np.mean(perm_betas_final)) if perm_betas_final.size > 0 else None,
        "perm_std": float(np.std(perm_betas_final)) if perm_betas_final.size > 0 else None,
        "p_emp": p_emp_final,
        "max_smd": max_smd,
        "iterations": iterations_final,
    }