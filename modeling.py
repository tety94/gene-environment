import statsmodels.formula.api as smf
import numpy as np
import pickle
from tqdm import tqdm
from config import TARGET_COL, MATCH_K, MIN_TREATED, MIN_SAMPLE_SIZE, N_PERM, RANDOM_STATE, MIN_OBS_COEF, N_PERM_HIGH, PVALUE_THRESHOLD
from db import save_variant_result, variant_already_done, get_conn, mark_variant_in_progress, reset_variant_in_progress
from matching import match_control_units, check_balance

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
    save_variant_result(conn, variant_original, muted,not_muted, None, None, None, 1, None, max_smd)

def process_single_variant(variant_col, variant_original, Ecols):
    df = pickle.load(open("temp_df.pkl", "rb"))
    conn = get_conn()

    # Segno come in progress
    if not mark_variant_in_progress(conn, variant_original):
        print(f"[INFO] Return: {variant_original} già in progress da un altro processo")
        conn.close()
        return None

    # Controllo numero di treated/control
    n_treated = int((df[variant_col] == 1).sum())
    n_control = int((df[variant_col] == 0).sum())
    if n_treated < MIN_TREATED or n_control == 0:
        print(f"[INFO] Return 3: {variant_original} numero insufficiente")
        save_variant_result_not_calculated(conn, variant_original, n_treated, n_control, None)
        conn.close()
        return None

    cols = [TARGET_COL, variant_col] + Ecols
    df_model = df[cols].dropna()
    if df_model.shape[0] < MIN_SAMPLE_SIZE:
        print(f"[INFO] Return 4: {variant_original} numero insufficiente")
        save_variant_result_not_calculated(conn, variant_original, n_treated, n_control, None)
        conn.close()
        return None

    cov_match = Ecols
    matched_obs = match_control_units(df_model, variant_col, k=MATCH_K, covariates_for_matching=cov_match)
    if matched_obs is None or matched_obs.shape[0] < MIN_SAMPLE_SIZE:
        print(f"[INFO] Return 5: {variant_original} numero insufficiente")
        save_variant_result_not_calculated(conn, variant_original, n_treated, n_control, None)
        conn.close()
        return None

    smd_results = check_balance(matched_obs, variant_col, cov_match)
    max_smd = max(smd_results.values()) if smd_results else 1
    if max_smd > 0.25:
        print(f"[WARN] Il matching per {variant_original} ha fallito il bilanciamento (Max SMD = {max_smd:.3f})")
        save_variant_result_not_calculated(conn, variant_original, n_treated, n_control, max_smd)
        conn.close()
        return variant_original

    # Costruzione formula e fit OLS
    formula = build_formula(TARGET_COL, variant_col, Ecols, [], matched_obs)
    mod = smf.ols(formula=formula, data=matched_obs).fit()
    interaction_name = _find_interaction_term(mod.params.index, variant_col)
    if interaction_name is None:
        conn.close()
        return None
    obs_coef = float(mod.params[interaction_name])

    # Controllo effetto minimo
    if obs_coef is None or abs(obs_coef) < MIN_OBS_COEF:
        save_variant_result(conn, variant_original,
                            int(matched_obs[variant_col].sum()),
                            int((matched_obs[variant_col] == 0).sum()),
                            obs_coef, None, None, 1, None, max_smd)
        conn.close()
        return variant_original

    rng = np.random.RandomState(RANDOM_STATE + (abs(hash(variant_col)) % 2_000_000))

    # ---------- PERMUTAZIONI LIGHT ----------
    perm_betas_light = []
    for _ in tqdm(range(N_PERM), desc=f"Perm test LIGHT {variant_col}", leave=False):
        df_perm = df_model.copy()
        df_perm[variant_col] = rng.permutation(df_perm[variant_col].values)
        matched_perm = match_control_units(df_perm, variant_col, k=MATCH_K, covariates_for_matching=cov_match)
        if matched_perm is None or matched_perm.shape[0] < MIN_SAMPLE_SIZE:
            perm_betas_light.append(np.nan)
            continue
        mod_perm = smf.ols(formula=formula, data=matched_perm).fit()
        perm_betas_light.append(mod_perm.params.get(interaction_name, np.nan))

    perm_betas_light = np.array([x for x in perm_betas_light if not np.isnan(x)])
    p_emp_light = float(np.mean(np.abs(perm_betas_light) >= np.abs(obs_coef))) if perm_betas_light.size > 0 else None

    # Salva risultato intermedio
    save_variant_result(conn, variant_original,
                        int(matched_obs[variant_col].sum()),
                        int((matched_obs[variant_col] == 0).sum()),
                        obs_coef,
                        float(np.mean(perm_betas_light)) if perm_betas_light.size > 0 else None,
                        float(np.std(perm_betas_light)) if perm_betas_light.size > 0 else None,
                        p_emp_light, N_PERM, max_smd
                        )

    # ---------- PERMUTAZIONI HIGH SOLO SE SIGNIFICATIVO ----------
    # Numero di permutazioni aggiuntive da calcolare
    n_additional = N_PERM_HIGH - N_PERM
    perm_betas_additional = []

    for _ in tqdm(range(n_additional), desc=f"Perm test ADD {variant_col}", leave=False):
        df_perm = df_model.copy()
        df_perm[variant_col] = rng.permutation(df_perm[variant_col].values)
        matched_perm = match_control_units(df_perm, variant_col, k=MATCH_K, covariates_for_matching=cov_match)
        if matched_perm is None or matched_perm.shape[0] < MIN_SAMPLE_SIZE:
            perm_betas_additional.append(np.nan)
            continue
        mod_perm = smf.ols(formula=formula, data=matched_perm).fit()
        perm_betas_additional.append(mod_perm.params.get(interaction_name, np.nan))

    # Combina i risultati con quelli light
    perm_betas_high = np.concatenate([perm_betas_light, perm_betas_additional])
    perm_betas_high = np.array([x for x in perm_betas_high if not np.isnan(x)])
    p_emp_high = float(np.mean(np.abs(perm_betas_high) >= np.abs(obs_coef))) if perm_betas_high.size > 0 else None

    # Aggiorna risultato finale
    save_variant_result(conn, variant_original,
                            int(matched_obs[variant_col].sum()),
                            int((matched_obs[variant_col] == 0).sum()),
                            obs_coef,
                            float(np.mean(perm_betas_high)) if perm_betas_high.size > 0 else None,
                            float(np.std(perm_betas_high)) if perm_betas_high.size > 0 else None,
                            p_emp_high, N_PERM_HIGH, max_smd
                        )

    # Segna la variante come completata
    reset_variant_in_progress(conn, variant_original, success=True)
    conn.close()

    return variant_original
