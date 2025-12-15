import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def _prepare_matching_matrix(df, cols):
    if not cols:
        raise ValueError("No columns provided for matching")
    features = []

    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(df[c].fillna(df[c].mean()))
        else:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
            features.append(dummies)

    if not features:
        raise ValueError("No valid matching features found in dataframe")

    X = pd.concat(features, axis=1)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled

def match_control_units(df, gene_col, k=2, covariates_for_matching=None):
    treated = df[df[gene_col] == 1].reset_index(drop=True)
    control = df[df[gene_col] == 0].reset_index(drop=True)

    if treated.shape[0] == 0 or control.shape[0] == 0:
        return None

    df_matching = pd.concat([treated, control], ignore_index=True)
    X = _prepare_matching_matrix(df_matching, covariates_for_matching)

    mask_t = df_matching[gene_col] == 1
    X_t = X[mask_t]
    X_c = X[~mask_t]

    if X_c.shape[0] == 0:
        return None

    k_used = min(k, X_c.shape[0])
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_c.values)
    distances, indices = nn.kneighbors(X_t.values)

    selected_ctrl_pos = np.unique(indices.flatten())
    ctrl_idx = X_c.index[selected_ctrl_pos]

    matched_controls = df_matching.loc[ctrl_idx]
    matched_treated = df_matching.loc[mask_t]
    matched_df = pd.concat([matched_treated, matched_controls], ignore_index=True)
    return matched_df

def check_balance(matched_df, gene_col, covariates_for_matching):
    if matched_df is None:
        return {}

    treated = matched_df[matched_df[gene_col] == 1]
    control = matched_df[matched_df[gene_col] == 0]
    smd_results = {}

    for c in covariates_for_matching:
        if c not in matched_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(matched_df[c]):
            c_treated = treated[c].fillna(treated[c].mean())
            c_control = control[c].fillna(control[c].mean())
            mean_t, mean_c = c_treated.mean(), c_control.mean()
            n_t, n_c = c_treated.count(), c_control.count()
            std_t, std_c = c_treated.std(ddof=1), c_control.std(ddof=1)
            pooled_std = np.sqrt(((n_t-1)*std_t**2 + (n_c-1)*std_c**2) / max(n_t+n_c-2,1))
            smd = 0 if pooled_std == 0 else abs(mean_t - mean_c) / pooled_std
            smd_results[c] = smd
        else:
            dummies = pd.get_dummies(matched_df[c].astype(str), prefix=c, drop_first=True)
            for d_col in dummies.columns:
                d_treated = dummies.loc[treated.index, d_col]
                d_control = dummies.loc[control.index, d_col]
                p_t, p_c = d_treated.mean(), d_control.mean()
                p_pooled = (p_t*len(d_treated) + p_c*len(d_control)) / (len(d_treated)+len(d_control))
                pooled_std = np.sqrt(p_pooled*(1-p_pooled))
                smd = 0 if pooled_std == 0 else abs(p_t - p_c) / pooled_std
                smd_results[d_col] = smd

    return smd_results
