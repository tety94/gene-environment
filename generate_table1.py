#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_table1.py
===================

Genera la "Tabella 1" di un paper (confronto Corte 1 vs Corte 2) a partire da:

  - componenti_ambientali_1.csv   -> dati corte 1 (demografia + variabili ambientali)
  - componenti_ambientali_2.csv   -> dati corte 2 (demografia + variabili ambientali)
  - componenti_ambientali_fumo.csv        -> fumo_boolean (id comuni a corte 1 e 2)
  - componenti_ambientali_scolarita.csv   -> education_level / education_years (id comuni)

Logica:
  1. Ogni corte viene arricchita (LEFT JOIN su "id") con le colonne *nuove*
     presenti nei file fumo/scolarita (le colonne demografiche duplicate come
     sex/onset_site/onset_age/diagnostic_delay vengono ignorate in fase di
     join per evitare conflitti, ma vengono controllate per coerenza).
  2. Le due corti arricchite vengono etichettate (Corte 1 / Corte 2) e concatenate.
  3. Per ogni variabile:
       - CONTINUA  -> mediana [IQR], test di Mann-Whitney U
       - CATEGORICA -> n (%), test Chi-quadro (o Fisher esatto se atteso <5)
  4. Output in cartella "output/":
       - table1.csv
       - table1.docx      (tabella pronta per il paper)
       - images/*.png     (boxplot, grafici a barre, trend p-value sui buffer)
       - run_log.txt       (log di join, missing, warning)

Uso:
    python generate_table1.py --input-dir . --output-dir output

Dipendenze:
    pip install pandas scipy matplotlib python-docx --break-system-packages
"""

import argparse
import os
import re
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

COHORT1_FILE = "componenti_ambientali_1.csv"
COHORT2_FILE = "componenti_ambientali_2.csv"
FUMO_FILE = "componenti_ambientali_fumo.csv"
SCOLARITA_FILE = "componenti_ambientali_scolarita.csv"

ID_COL = "id"

# File genotipici che definiscono l'appartenenza reale alle due corti
# (l'id in componenti_ambientali_1/2.csv corrisponde ESATTAMENTE, con
# duplicazione, all'id in questi file: es. "RES02977_RES02977").
# Corte 1 = id presenti nel file gen1 (RES...), Corte 2 = id presenti nel file
# gen2 (ACH...). Se un id compare in entrambi, vince Corte 1.
GEN1_IDS_FILE_DEFAULT = "/mnt/cresla_prod/genome_datasets/merged_csv/full_chr_gen1_test1.csv"
GEN2_IDS_FILE_DEFAULT = "/srv/python-projects/gene-environment/output_combined_risaie.csv"

# Colonna calcolata usata per il confronto in Tabella 1 ("Corte 1"/"Corte 2"),
# derivata dall'appartenenza ai file genotipici sopra, NON da parals_codals
# (che è solo un'etichetta arbitraria e viene trattata come normale variabile
# categorica descrittiva).
GROUP_COL = "corte"

# Variabili demografiche/cliniche continue attese nei file corte (se presenti)
BASE_CONTINUOUS_VARS = ["onset_age", "diagnostic_delay", "survival"]

# Variabili demografiche/cliniche categoriche attese nei file corte (se presenti)
BASE_CATEGORICAL_VARS = ["sex", "onset_site", "parals_codals"]

# Colonne "demografiche" duplicate nei file fumo/scolarita: NON vengono importate
# in join (si tiene la versione presente nel file di corte), ma vengono
# confrontate per coerenza e loggate se discordanti.
DEMOGRAPHIC_OVERLAP_COLS = ["sex", "onset_site", "onset_age", "diagnostic_delay"]

# Pattern per individuare automaticamente le variabili ambientali (6 buffer x 3 categorie)
BUFFER_VAR_PATTERN = re.compile(r"^(seminativi|vigneti|risaie)_(\d+)$")

ALPHA = 0.05


def load_ids_from_genotype_file(path, log, label):
    """Legge SOLO il primo campo (id) di ogni riga di un file genotipico,
    senza parsare le migliaia di colonne SNP successive. Efficiente anche su
    file molto grandi perché si ferma alla prima virgola di ogni riga."""
    if not os.path.exists(path):
        log.add(f"ERRORE: file genotipico non trovato per {label} -> {path}")
        return set()
    ids = set()
    n_lines = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()  # header
        for line in f:
            n_lines += 1
            comma_idx = line.find(",")
            id_val = line[:comma_idx].strip() if comma_idx != -1 else line.strip()
            if id_val:
                ids.add(id_val)
    log.add(f"File genotipico {label} ({path}): {n_lines} righe, {len(ids)} id unici")
    return ids


# ----------------------------------------------------------------------------
# UTILITY
# ----------------------------------------------------------------------------

class RunLog:
    """Colleziona messaggi da scrivere poi su run_log.txt"""

    def __init__(self):
        self.lines = []

    def add(self, msg):
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line)
        self.lines.append(line)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))


def read_csv_robust(path, log):
    """Legge un csv gestendo automaticamente virgolette e incoraggiando
    la conversione numerica delle colonne che lo consentono."""
    if not os.path.exists(path):
        log.add(f"ERRORE: file non trovato -> {path}")
        return None
    df = pd.read_csv(path, sep=",", quotechar='"', skipinitialspace=True)
    df.columns = [c.strip().strip('"') for c in df.columns]
    if ID_COL not in df.columns:
        log.add(f"ATTENZIONE: colonna '{ID_COL}' non trovata in {path} (colonne: {list(df.columns)})")
    else:
        df[ID_COL] = df[ID_COL].astype(str).str.strip()
    log.add(f"Letto {path}: {df.shape[0]} righe, {df.shape[1]} colonne")
    return df


def detect_buffer_vars(df):
    """Ritorna la lista delle colonne ambientali seminativi/vigneti/risaie_<buffer>
    trovate nel dataframe, ordinate per categoria e per distanza buffer."""
    found = [c for c in df.columns if BUFFER_VAR_PATTERN.match(c)]

    def sort_key(c):
        m = BUFFER_VAR_PATTERN.match(c)
        return (m.group(1), int(m.group(2)))

    return sorted(found, key=sort_key)


def check_demographic_consistency(cohort_df, side_df, side_name, cohort_name, log):
    """Confronta le colonne demografiche in comune (per stesso id) fra il file
    di corte e il file fumo/scolarita, e logga eventuali discrepanze."""
    common_cols = [c for c in DEMOGRAPHIC_OVERLAP_COLS if c in cohort_df.columns and c in side_df.columns]
    if not common_cols:
        return
    merged = cohort_df[[ID_COL] + common_cols].merge(
        side_df[[ID_COL] + common_cols], on=ID_COL, how="inner", suffixes=("_corte", f"_{side_name}")
    )
    n_mismatch_total = 0
    for col in common_cols:
        c1 = merged[f"{col}_corte"]
        c2 = merged[f"{col}_{side_name}"]
        mismatch = merged[(c1.astype(str) != c2.astype(str)) & c1.notna() & c2.notna()]
        if len(mismatch) > 0:
            n_mismatch_total += len(mismatch)
            log.add(
                f"ATTENZIONE [{cohort_name} vs {side_name}]: {len(mismatch)} id con valori "
                f"discordanti per '{col}' (mantenuto il valore del file di corte)."
            )
    if n_mismatch_total == 0:
        log.add(f"OK: nessuna discrepanza demografica fra {cohort_name} e {side_name} sulle colonne {common_cols}")


def merge_side_file(cohort_df, side_df, side_name, cohort_name, log):
    """Left-join di cohort_df con le sole colonne NUOVE presenti in side_df
    (tutto tranne l'id e le colonne demografiche già presenti in cohort_df)."""
    if side_df is None:
        return cohort_df

    check_demographic_consistency(cohort_df, side_df, side_name, cohort_name, log)

    new_cols = [c for c in side_df.columns if c not in cohort_df.columns and c != ID_COL]
    if not new_cols:
        log.add(f"ATTENZIONE: nessuna colonna nuova da importare da {side_name} in {cohort_name}")
        return cohort_df

    slim = side_df[[ID_COL] + new_cols].drop_duplicates(subset=ID_COL)
    before = cohort_df.shape[0]
    merged = cohort_df.merge(slim, on=ID_COL, how="left")
    n_missing = merged[new_cols[0]].isna().sum()
    log.add(
        f"Join {cohort_name} + {side_name}: {before} righe -> {merged.shape[0]} righe; "
        f"colonne aggiunte: {new_cols}; id senza corrispondenza in {side_name}: {n_missing}"
    )
    return merged


# ----------------------------------------------------------------------------
# STATISTICA
# ----------------------------------------------------------------------------

def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n.d."
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def continuous_summary(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return "n=0", n
    med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
    return f"{med:.1f} [{q1:.1f}\u2013{q3:.1f}]", n


def mannwhitney_p(s1, s2):
    s1 = pd.to_numeric(s1, errors="coerce").dropna()
    s2 = pd.to_numeric(s2, errors="coerce").dropna()
    if len(s1) < 1 or len(s2) < 1:
        return np.nan
    try:
        _, p = mannwhitneyu(s1, s2, alternative="two-sided")
        return p
    except ValueError:
        return np.nan


def categorical_counts(series):
    s = series.dropna()
    counts = s.value_counts()
    total = counts.sum()
    return counts, total


def chi2_or_fisher_p(df, var, group_col):
    ct = pd.crosstab(df[var], df[group_col])
    ct = ct.loc[:, (ct.sum(axis=0) > 0)]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return "n.d.", np.nan
    try:
        chi2, p, dof, expected = chi2_contingency(ct, correction=False)
    except ValueError:
        return "n.d.", np.nan
    if ct.shape == (2, 2):
        if (expected < 5).any():
            try:
                _, p = fisher_exact(ct.values)
                return "Fisher esatto", p
            except ValueError:
                return "n.d.", np.nan
        return "Chi-quadro", p
    test_name = "Chi-quadro"
    if (expected < 5).sum() > 0:
        test_name = "Chi-quadro (attenzione: celle attese <5)"
    return test_name, p


# ----------------------------------------------------------------------------
# COSTRUZIONE TABELLA 1
# ----------------------------------------------------------------------------

def build_table1(df, group_col, continuous_vars, categorical_vars, log):
    """Ritorna una lista di dict, una riga per ogni riga di tabella
    (header di variabile + eventuali sotto-righe categoria)."""
    groups = sorted(df[group_col].dropna().unique())
    if len(groups) != 2:
        log.add(f"ATTENZIONE: attesi 2 gruppi in '{group_col}', trovati: {groups}")
    g1, g2 = groups[0], groups[1]

    rows = []
    n1_tot = (df[group_col] == g1).sum()
    n2_tot = (df[group_col] == g2).sum()
    rows.append({
        "Variabile": "N totale, n",
        "Categoria": "",
        f"{g1}": f"{n1_tot}",
        f"{g2}": f"{n2_tot}",
        "p-value": "",
        "Test": "",
        "significativo": False,
    })

    for var in continuous_vars:
        if var not in df.columns:
            log.add(f"ATTENZIONE: variabile continua '{var}' non presente, saltata")
            continue
        s1 = df.loc[df[group_col] == g1, var]
        s2 = df.loc[df[group_col] == g2, var]
        summ1, n1 = continuous_summary(s1)
        summ2, n2 = continuous_summary(s2)
        p = mannwhitney_p(s1, s2)
        rows.append({
            "Variabile": f"{var}, mediana [IQR]",
            "Categoria": "",
            f"{g1}": f"{summ1} (n={n1})",
            f"{g2}": f"{summ2} (n={n2})",
            "p-value": fmt_p(p),
            "Test": "Mann-Whitney U",
            "significativo": (p is not None and not np.isnan(p) and p < ALPHA),
        })

    for var in categorical_vars:
        if var not in df.columns:
            log.add(f"ATTENZIONE: variabile categorica '{var}' non presente, saltata")
            continue
        c1, tot1 = categorical_counts(df.loc[df[group_col] == g1, var])
        c2, tot2 = categorical_counts(df.loc[df[group_col] == g2, var])
        test_name, p = chi2_or_fisher_p(df, var, group_col)
        sig = (p is not None and not np.isnan(p) and p < ALPHA)
        rows.append({
            "Variabile": f"{var}, n (%)",
            "Categoria": "",
            f"{g1}": f"n={tot1}",
            f"{g2}": f"n={tot2}",
            "p-value": fmt_p(p),
            "Test": test_name,
            "significativo": sig,
        })
        categories = sorted(set(c1.index.tolist()) | set(c2.index.tolist()), key=lambda x: str(x))
        for cat in categories:
            n1 = c1.get(cat, 0)
            n2 = c2.get(cat, 0)
            pct1 = (n1 / tot1 * 100) if tot1 else 0
            pct2 = (n2 / tot2 * 100) if tot2 else 0
            rows.append({
                "Variabile": "",
                "Categoria": f"   {cat}",
                f"{g1}": f"{n1} ({pct1:.1f}%)",
                f"{g2}": f"{n2} ({pct2:.1f}%)",
                "p-value": "",
                "Test": "",
                "significativo": False,
            })

    return rows, g1, g2


# ----------------------------------------------------------------------------
# EXPORT CSV
# ----------------------------------------------------------------------------

def save_table1_csv(rows, g1, g2, path):
    df_out = pd.DataFrame(rows)
    cols = ["Variabile", "Categoria", str(g1), str(g2), "p-value", "Test"]
    cols = [c for c in cols if c in df_out.columns]
    df_out = df_out[cols]
    df_out.to_csv(path, index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------------
# EXPORT DOCX
# ----------------------------------------------------------------------------

def _set_cell_shading(cell, color_hex):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color_hex)
    cell._tc.get_or_add_tcPr().append(shd)


def save_table1_docx(rows, g1, g2, path, images_for_appendix=None):
    doc = Document()

    title = doc.add_heading("Tabella 1. Caratteristiche demografiche, cliniche e ambientali "
                             "per corte di studio", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    subtitle = doc.add_paragraph()
    run = subtitle.add_run(
        f"Confronto {g1} vs {g2}. Variabili continue: mediana [IQR], test di Mann-Whitney U. "
        f"Variabili categoriche: n (%), test Chi-quadro o Fisher esatto (celle attese <5). "
        f"p < {ALPHA} evidenziato in grassetto."
    )
    run.italic = True
    run.font.size = Pt(9)

    headers = ["Variabile", str(g1), str(g2), "p-value", "Test"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
        _set_cell_shading(hdr_cells[i], "D9E2F3")

    for row in rows:
        cells = table.add_row().cells
        var_label = row["Variabile"] if row["Variabile"] else row["Categoria"]
        values = [var_label, row.get(str(g1), ""), row.get(str(g2), ""), row["p-value"], row["Test"]]
        for i, v in enumerate(values):
            cells[i].text = str(v)
            if row.get("significativo") and i == 3:
                for p in cells[i].paragraphs:
                    for r in p.runs:
                        r.bold = True
                        r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
            if not row["Variabile"] and row["Categoria"] and i == 0:
                for p in cells[i].paragraphs:
                    for r in p.runs:
                        r.italic = True

    for col_idx, width in enumerate([Inches(2.3), Inches(1.5), Inches(1.5), Inches(0.9), Inches(1.6)]):
        for row in table.rows:
            row.cells[col_idx].width = width

    if images_for_appendix:
        doc.add_page_break()
        doc.add_heading("Appendice - Figure", level=1)
        for img_path, caption in images_for_appendix:
            if os.path.exists(img_path):
                doc.add_picture(img_path, width=Inches(6))
                cap = doc.add_paragraph(caption)
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in cap.runs:
                    r.italic = True
                    r.font.size = Pt(9)

    doc.save(path)


# ----------------------------------------------------------------------------
# IMMAGINI
# ----------------------------------------------------------------------------

def plot_continuous_box(df, var, group_col, g1, g2, out_path, log):
    s1 = pd.to_numeric(df.loc[df[group_col] == g1, var], errors="coerce").dropna()
    s2 = pd.to_numeric(df.loc[df[group_col] == g2, var], errors="coerce").dropna()
    if len(s1) == 0 or len(s2) == 0:
        return None
    p = mannwhitney_p(s1, s2)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    try:
        bp = ax.boxplot([s1, s2], tick_labels=[str(g1), str(g2)], patch_artist=True, showmeans=True)
    except TypeError:
        bp = ax.boxplot([s1, s2], labels=[str(g1), str(g2)], patch_artist=True, showmeans=True)
    colors = ["#4C72B0", "#DD8452"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_title(f"{var}\nMann-Whitney U p={fmt_p(p)}")
    ax.set_ylabel(var)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_buffer_grouped_box(df, category, group_col, g1, g2, out_dir, log):
    """category in {'seminativi','vigneti','risaie'}: boxplot raggruppato per
    distanza buffer, affiancando corte1 e corte2."""
    buffer_cols = sorted(
        [c for c in df.columns if c.startswith(category + "_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if not buffer_cols:
        return None
    distances = [c.split("_")[1] for c in buffer_cols]

    fig, ax = plt.subplots(figsize=(max(6, len(buffer_cols) * 1.1), 5))
    positions1, positions2 = [], []
    data1, data2 = [], []
    width = 0.35
    for i, col in enumerate(buffer_cols):
        s1 = pd.to_numeric(df.loc[df[group_col] == g1, col], errors="coerce").dropna()
        s2 = pd.to_numeric(df.loc[df[group_col] == g2, col], errors="coerce").dropna()
        data1.append(s1)
        data2.append(s2)
        positions1.append(i - width / 2)
        positions2.append(i + width / 2)

    bp1 = ax.boxplot(data1, positions=positions1, widths=width, patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data2, positions=positions2, widths=width, patch_artist=True, showfliers=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#DD8452")
        patch.set_alpha(0.6)

    ax.set_xticks(range(len(buffer_cols)))
    ax.set_xticklabels([f"{d}m" for d in distances])
    ax.set_xlabel("Buffer (metri)")
    ax.set_ylabel(f"{category} (valore)")
    ax.set_title(f"{category}: confronto {g1} vs {g2} per buffer")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], [str(g1), str(g2)], loc="best")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"boxplot_{category}_per_buffer.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_buffer_pvalue_trend(df, category, group_col, g1, g2, out_dir, log):
    buffer_cols = sorted(
        [c for c in df.columns if c.startswith(category + "_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if not buffer_cols:
        return None
    distances, pvals = [], []
    for col in buffer_cols:
        s1 = df.loc[df[group_col] == g1, col]
        s2 = df.loc[df[group_col] == g2, col]
        p = mannwhitney_p(s1, s2)
        distances.append(int(col.split("_")[1]))
        pvals.append(p)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(distances, pvals, marker="o", color="#4C72B0")
    ax.axhline(ALPHA, color="red", linestyle="--", linewidth=1, label=f"soglia p={ALPHA}")
    ax.set_xlabel("Buffer (metri)")
    ax.set_ylabel("p-value (Mann-Whitney U)")
    ax.set_title(f"{category}: andamento del p-value al variare del buffer")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"pvalue_trend_{category}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_categorical_bar(df, var, group_col, g1, g2, out_dir, log):
    if var not in df.columns:
        return None
    c1, tot1 = categorical_counts(df.loc[df[group_col] == g1, var])
    c2, tot2 = categorical_counts(df.loc[df[group_col] == g2, var])
    categories = sorted(set(c1.index.tolist()) | set(c2.index.tolist()), key=lambda x: str(x))
    if not categories:
        return None
    pct1 = [c1.get(cat, 0) / tot1 * 100 if tot1 else 0 for cat in categories]
    pct2 = [c2.get(cat, 0) / tot2 * 100 if tot2 else 0 for cat in categories]

    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(5, len(categories) * 1.2), 4.5))
    ax.bar(x - width / 2, pct1, width, label=str(g1), color="#4C72B0", alpha=0.8)
    ax.bar(x + width / 2, pct2, width, label=str(g2), color="#DD8452", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in categories], rotation=0)
    ax.set_ylabel("% entro la corte")
    ax.set_title(f"{var}: distribuzione per corte")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"barplot_{var}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Genera Tabella 1 (Corte 1 vs Corte 2) con test statistici e figure")
    parser.add_argument("--input-dir", default=".", help="Cartella con i 4 csv ambientali/demografici in input")
    parser.add_argument("--output-dir", default="output", help="Cartella di output")
    parser.add_argument("--gen1-ids-file", default=GEN1_IDS_FILE_DEFAULT,
                         help="File genotipico i cui id definiscono la Corte 1 (RES...)")
    parser.add_argument("--gen2-ids-file", default=GEN2_IDS_FILE_DEFAULT,
                         help="File genotipico i cui id definiscono la Corte 2 (ACH...)")
    args = parser.parse_args()

    out_dir = args.output_dir
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    log = RunLog()
    log.add("Avvio generazione Tabella 1")

    cohort1 = read_csv_robust(os.path.join(args.input_dir, COHORT1_FILE), log)
    cohort2 = read_csv_robust(os.path.join(args.input_dir, COHORT2_FILE), log)
    fumo = read_csv_robust(os.path.join(args.input_dir, FUMO_FILE), log)
    scolarita = read_csv_robust(os.path.join(args.input_dir, SCOLARITA_FILE), log)

    if cohort1 is None or cohort2 is None:
        log.add("ERRORE FATALE: mancano i file delle corti, impossibile proseguire")
        log.save(os.path.join(out_dir, "run_log.txt"))
        sys.exit(1)

    # componenti_ambientali_1 e _2 vengono unite (contengono lo stesso set di
    # pazienti/colonne); se un id compare in entrambi si tiene una sola riga.
    raw_combined = pd.concat([cohort1, cohort2], ignore_index=True, sort=False)
    n_before = raw_combined.shape[0]
    raw_combined = raw_combined.drop_duplicates(subset=ID_COL, keep="first")
    n_after = raw_combined.shape[0]
    if n_before != n_after:
        log.add(f"Uniti componenti_ambientali_1 e _2: {n_before} righe -> {n_after} righe dopo deduplica su id "
                 f"({n_before - n_after} duplicati rimossi)")
    else:
        log.add(f"Uniti componenti_ambientali_1 e _2: {n_after} righe, nessun id duplicato trovato")

    full = merge_side_file(raw_combined, fumo, "fumo", "dataset combinato", log)
    full = merge_side_file(full, scolarita, "scolarita", "dataset combinato", log)

    # ---------------- ASSEGNAZIONE CORTE DA FILE GENOTIPICI ----------------
    gen1_ids = load_ids_from_genotype_file(args.gen1_ids_file, log, "gen1 (Corte 1, RES)")
    gen2_ids = load_ids_from_genotype_file(args.gen2_ids_file, log, "gen2 (Corte 2, ACH)")

    overlap = gen1_ids & gen2_ids
    if overlap:
        log.add(f"ATTENZIONE: {len(overlap)} id presenti in ENTRAMBI i file genotipici; "
                 f"assegnati a Corte 1 per priorità.")

    def assign_cohort(id_val):
        doubled = f"{id_val}_{id_val}"
        if id_val in gen1_ids or doubled in gen1_ids:
            return "Corte 1"
        if id_val in gen2_ids or doubled in gen2_ids:
            return "Corte 2"
        return None

    full[GROUP_COL] = full[ID_COL].astype(str).map(assign_cohort)

    n_unassigned = full[GROUP_COL].isna().sum()
    if n_unassigned > 0:
        log.add(f"ATTENZIONE: {n_unassigned} id in componenti_ambientali non trovati in NESSUNO dei due file "
                 f"genotipici; verranno esclusi dall'analisi. Controlla il formato dell'id "
                 f"(atteso identico a quello genotipico, es. 'RES02977_RES02977').")
        full = full.dropna(subset=[GROUP_COL])

    log.add(f"Dataset finale: {full.shape[0]} righe, {full.shape[1]} colonne. "
            f"Distribuzione '{GROUP_COL}': {full[GROUP_COL].value_counts().to_dict()}")

    buffer_vars = detect_buffer_vars(full)
    continuous_vars = [v for v in BASE_CONTINUOUS_VARS if v in full.columns]
    if "education_years" in full.columns:
        continuous_vars.append("education_years")
    continuous_vars += buffer_vars

    categorical_vars = [v for v in BASE_CATEGORICAL_VARS if v in full.columns and v != GROUP_COL]
    if "fumo_boolean" in full.columns:
        categorical_vars.append("fumo_boolean")
    if "education_level" in full.columns:
        categorical_vars.append("education_level")

    log.add(f"Variabili continue analizzate ({len(continuous_vars)}): {continuous_vars}")
    log.add(f"Variabili categoriche analizzate ({len(categorical_vars)}): {categorical_vars}")

    rows, g1, g2 = build_table1(full, GROUP_COL, continuous_vars, categorical_vars, log)

    csv_path = os.path.join(out_dir, "table1.csv")
    save_table1_csv(rows, g1, g2, csv_path)
    log.add(f"Salvato {csv_path}")

    full.to_csv(os.path.join(out_dir, "dataset_combinato.csv"), index=False, encoding="utf-8-sig")
    log.add("Salvato dataset_combinato.csv (join completo, utile per analisi successive)")

    # ---------------- IMMAGINI ----------------
    appendix_images = []

    for var in [v for v in BASE_CONTINUOUS_VARS + (["education_years"] if "education_years" in full.columns else []) if v in full.columns]:
        p = plot_continuous_box(full, var, GROUP_COL, g1, g2, os.path.join(img_dir, f"boxplot_{var}.png"), log)
        if p:
            appendix_images.append((p, f"Distribuzione di {var} per corte"))

    for var in categorical_vars:
        p = plot_categorical_bar(full, var, GROUP_COL, g1, g2, img_dir, log)
        if p:
            appendix_images.append((p, f"Distribuzione di {var} per corte"))

    for category in ["seminativi", "vigneti", "risaie"]:
        p1 = plot_buffer_grouped_box(full, category, GROUP_COL, g1, g2, img_dir, log)
        if p1:
            appendix_images.append((p1, f"{category}: boxplot per buffer, {g1} vs {g2}"))
        p2 = plot_buffer_pvalue_trend(full, category, GROUP_COL, g1, g2, img_dir, log)
        if p2:
            appendix_images.append((p2, f"{category}: andamento del p-value al variare del buffer"))

    log.add(f"Generate {len(appendix_images)} immagini in {img_dir}")

    # ---------------- DOCX ----------------
    docx_path = os.path.join(out_dir, "table1.docx")
    save_table1_docx(rows, g1, g2, docx_path, images_for_appendix=appendix_images)
    log.add(f"Salvato {docx_path}")

    log.save(os.path.join(out_dir, "run_log.txt"))
    log.add("Completato.")


if __name__ == "__main__":
    main()