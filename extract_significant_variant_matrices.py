#!/usr/bin/env python3
"""
Estrae, per le varianti significative (secondo db.get_significant_results()),
il genotipo binario (0 = non mutato, 1 = mutato) per ciascun paziente in
gen1, gen2 e gen3, direttamente dai VCF indicizzati per cromosoma (bcftools),
invece che dai CSV "full genome" derivati (lenti, e nel caso di gen2/gen3
con colonne duplicate per lo stesso sito, vedi diagnose_variant_structure.py).

Vantaggi rispetto all'approccio CSV:
  - Query di regione dirette via indice .tbi -> nessuna scansione dell'intero
    file, il tempo di estrazione e' indipendente dalla dimensione del VCF.
  - Nessun problema di colonne duplicate: si legge direttamente dal VCF
    sorgente, non da un CSV derivato con una conversione a monte sconosciuta.
  - Gestione corretta di siti multiallelici: si verifica che REF/ALT nel VCF
    corrispondano esattamente a quanto atteso, e si calcola il dosaggio
    sull'indice allelico corretto (non si assume sempre "ALT singolo").

Fonte varianti significative : db.get_significant_results()
Fonte genotipi                : VCF indicizzati per cromosoma:
    gen1: /mnt/cresla_prod/genome_datasets/gen1/gen1_onlycases_vcf_chr{N}.vcf.gz
    gen2: /mnt/cresla_prod/genome_datasets/gen2/gen2_vcf_chr{N}.vcf.gz
    gen3: /mnt/cresla_prod/genome_datasets/gen3/gen3_vcf_chr{N}.vcf.gz

NOTA IMPORTANTE: gen1 ("onlycases") contiene SOLO pazienti caso, mentre
gen2/gen3 hanno una composizione diversa (confermato dall'utente) -> le tre
coorti non sono direttamente comparabili come popolazione, la colonna
"generation" nell'output serve proprio a poterle distinguere/filtrare a valle.

Output: un UNICO CSV combinato con:
  - colonna "id"          : id paziente prefissato per coorte (gen1_..., gen2_..., gen3_...)
  - colonna "generation"   : 1, 2, o 3
  - una colonna per ciascuna variante significativa (0/1/NA)

Usage: python extract_significant_variants_vcf.py
"""
import os
import re
import time
import logging
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

from db import get_significant_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────
VCF_DIR = {
    1: "/mnt/cresla_prod/genome_datasets/gen1",
    2: "/mnt/cresla_prod/genome_datasets/gen2",
    3: "/mnt/cresla_prod/genome_datasets/gen3",
}
VCF_FILENAME_PATTERN = {
    1: "gen1_onlycases_vcf_chr{chrom}.vcf.gz",
    2: "gen2_vcf_chr{chrom}.vcf.gz",
    3: "gen3_vcf_chr{chrom}.vcf.gz",
}
OUT_DIR = "/srv/python-projects/gene-environment/significant_variant_matrices"
BCFTOOLS = "bcftools"
# ─────────────────────────────────────────────────────────────────────────────


def build_variant_label(chromosome, position, mutation):
    return f"{chromosome}_{position}_{mutation}"


def vcf_path_for(generation, chrom):
    filename = VCF_FILENAME_PATTERN[generation].format(chrom=chrom)
    return os.path.join(VCF_DIR[generation], filename)


def detect_chrom_naming(vcf_path):
    """
    Determina se il VCF usa 'chr1' o '1' come nome di contig, leggendo SOLO
    l'header (##contig=<ID=...>), senza scansionare i dati. Ritorna il primo
    ID di contig trovato nell'header, o None se non trovato (fallback: si
    prova comunque con il nome cromosoma cosi' com'e').
    """
    try:
        result = subprocess.run(
            [BCFTOOLS, "view", "-h", vcf_path],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        log.warning("Impossibile leggere l'header di %s: %s", vcf_path, e)
        return None

    for line in result.stdout.splitlines():
        if line.startswith("##contig"):
            m = re.search(r"ID=([^,>]+)", line)
            if m:
                return m.group(1)
    return None


def resolve_chrom_name(chrom_raw, vcf_path, _cache={}):
    """
    Ritorna il nome di cromosoma da usare nella query (es. '12' o 'chr12'),
    rilevato una sola volta per file e messo in cache.
    """
    if vcf_path in _cache:
        contig_example = _cache[vcf_path]
    else:
        contig_example = detect_chrom_naming(vcf_path)
        _cache[vcf_path] = contig_example

    if contig_example is None:
        return chrom_raw  # fallback: nessuna informazione, si prova cosi' com'e'

    if contig_example.startswith("chr") and not chrom_raw.startswith("chr"):
        return f"chr{chrom_raw}"
    if not contig_example.startswith("chr") and chrom_raw.startswith("chr"):
        return chrom_raw[3:]
    return chrom_raw


def get_samples(vcf_path):
    result = subprocess.run(
        [BCFTOOLS, "query", "-l", vcf_path],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip().split("\n") if result.stdout.strip() else []


def query_positions(vcf_path, chrom, positions):
    """
    Interroga il VCF per una lista di posizioni sullo stesso cromosoma,
    sfruttando l'indice .tbi (query di regione, non scansione completa).
    Ritorna le righe grezze di bcftools query (CHROM, POS, REF, ALT, GT...).
    """
    region_list = ",".join(f"{chrom}:{pos}-{pos}" for pos in positions)
    cmd = [
        BCFTOOLS, "query",
        "-r", region_list,
        "-f", "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n",
        vcf_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def parse_gt_row(line, target_ref, target_alt):
    """
    Ritorna la lista di dosaggi (0/1/2/-1=missing) per la riga, oppure None
    se REF/ALT nel VCF non corrispondono a quanto atteso (mismatch da
    segnalare, non da ignorare silenziosamente).
    """
    parts = line.split("\t")
    ref, alt_field = parts[2], parts[3]
    gts = parts[4:]

    alt_list = alt_field.split(",")
    if ref != target_ref or target_alt not in alt_list:
        return None

    target_idx = str(alt_list.index(target_alt) + 1)  # 0=REF, 1=primo ALT, ecc.

    dosages = []
    for gt in gts:
        alleles = re.split(r"[/|]", gt)
        if any(a == "." for a in alleles):
            dosages.append(-1)
        else:
            dosages.append(sum(1 for a in alleles if a == target_idx))
    return dosages


def binarize_vectorized(df_dosage):
    """
    0 -> 0 (non mutato), 1/2 -> 1 (mutato), -1/altro/NaN -> NaN (missing).
    Vettorizzato con numpy invece di applymap (piu' veloce, e applymap e'
    deprecato da pandas >=2.1).
    """
    df_num = df_dosage.apply(pd.to_numeric, errors="coerce")
    arr = np.select(
        [df_num.eq(0), df_num.isin([1, 2])],
        [0, 1],
        default=np.nan,
    )
    return pd.DataFrame(arr, index=df_dosage.index, columns=df_dosage.columns)


def extract_generation(generation, variants_df):
    """
    variants_df: DataFrame con colonne chrom, pos, ref, alt, label per TUTTE
    le varianti significative (stesse per ogni generazione, si controlla per
    ciascuna se e' presente nel VCF di questa generazione).

    Ritorna (df_bin, labels_not_found) dove df_bin ha indice = id paziente
    (senza prefisso, aggiunto dal chiamante) e colonne = varianti trovate.
    """
    t_start = time.perf_counter()
    log.info("=== Generazione %d: inizio ===", generation)

    all_dosages = {}  # label -> {sample_id: dosage}
    labels_not_found = []
    samples_by_chrom = {}

    for chrom, group in variants_df.groupby("chrom"):
        vcf_path = vcf_path_for(generation, chrom)
        if not os.path.exists(vcf_path):
            log.warning("  VCF non trovato per generazione %d, chr%s: %s", generation, chrom, vcf_path)
            labels_not_found.extend(group["label"].tolist())
            continue

        chrom_name = resolve_chrom_name(str(chrom), vcf_path)
        samples = get_samples(vcf_path)
        samples_by_chrom[chrom] = samples

        positions = group["pos"].tolist()
        t0 = time.perf_counter()
        rows = query_positions(vcf_path, chrom_name, positions)
        log.info("  chr%s: %d varianti richieste, %d righe trovate nel VCF (%.2fs)",
                  chrom, len(positions), len(rows), time.perf_counter() - t0)

        # indicizza le righe trovate per posizione (potrebbero essercene piu' di
        # una per posizione in caso di siti multiallelici non normalizzati)
        rows_by_pos = defaultdict(list)
        for line in rows:
            pos_found = int(line.split("\t")[1])
            rows_by_pos[pos_found].append(line)

        for _, variant in group.iterrows():
            label = variant["label"]
            pos = variant["pos"]
            ref = variant["ref"]
            alt = variant["alt"]

            candidate_rows = rows_by_pos.get(pos, [])
            dosages = None
            for line in candidate_rows:
                dosages = parse_gt_row(line, ref, alt)
                if dosages is not None:
                    break

            if dosages is None:
                if candidate_rows:
                    log.warning(
                        "  [MISMATCH] %s: trovata posizione ma REF/ALT non corrispondono "
                        "(atteso %s>%s). Riga/e nel VCF: %s",
                        label, ref, alt, candidate_rows,
                    )
                labels_not_found.append(label)
                continue

            all_dosages[label] = dict(zip(samples, dosages))

    if not all_dosages:
        log.error("Generazione %d: nessuna variante estratta.", generation)
        return pd.DataFrame(), labels_not_found

    df = pd.DataFrame(all_dosages)  # indice = sample id (union implicita), colonne = varianti trovate
    df_bin = binarize_vectorized(df)

    log.info("=== Generazione %d: completata in %.1fs (%d pazienti, %d varianti trovate) ===",
              generation, time.perf_counter() - t_start, len(df_bin), df_bin.shape[1])
    return df_bin, labels_not_found


def main():
    log.info("=== Recupero varianti significative dal DB ===")
    t_total = time.perf_counter()

    sig = get_significant_results()
    if sig.empty:
        log.info("Nessuna variante significativa trovata. Esco.")
        return

    sig["ref"] = sig["mutation"].apply(lambda m: m.split("_", 1)[0])
    sig["alt"] = sig["mutation"].apply(lambda m: m.split("_", 1)[1])
    sig["chrom"] = sig["chromosome"].astype(str)
    sig["pos"] = sig["position"].astype(int)
    sig["label"] = sig.apply(
        lambda r: build_variant_label(r["chromosome"], r["position"], r["mutation"]), axis=1
    )
    variants_df = sig[["chrom", "pos", "ref", "alt", "label"]].drop_duplicates(subset="label")
    log.info("%d varianti uniche da estrarre.", len(variants_df))

    os.makedirs(OUT_DIR, exist_ok=True)

    all_variant_labels = sorted(variants_df["label"].unique())
    combined_parts = []

    for gen in (1, 2, 3):
        df_bin, not_found = extract_generation(gen, variants_df)
        if not_found:
            log.warning("Generazione %d: %d/%d varianti non trovate/non estratte: %s",
                        gen, len(not_found), len(all_variant_labels), not_found)

        if df_bin.empty:
            continue

        # riallinea alle 34 varianti totali (colonne mancanti -> NaN), cosi'
        # il file combinato ha sempre lo stesso schema di colonne per ogni coorte
        df_bin = df_bin.reindex(columns=all_variant_labels)

        # id prefissato per coorte + colonna generation esplicita
        df_bin.index = [f"gen{gen}_{sample_id}" for sample_id in df_bin.index]
        df_bin.index.name = "id"
        df_bin.insert(0, "generation", gen)

        combined_parts.append(df_bin)

    if not combined_parts:
        log.error("Nessuna generazione ha prodotto dati. Esco senza scrivere output.")
        return

    combined = pd.concat(combined_parts, axis=0)

    out_path = os.path.join(OUT_DIR, "combined_significant_variants.csv")
    combined.to_csv(out_path)

    log.info("-> Scritto %s (%d pazienti totali, %d varianti, coorti: %s)",
              out_path, len(combined), len(all_variant_labels),
              combined["generation"].value_counts().to_dict())
    log.info("=== Fatto in %.1fs totali ===", time.perf_counter() - t_total)


if __name__ == "__main__":
    main()