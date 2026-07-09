#!/usr/bin/env python3
"""
Estrae, per le varianti significative in ENTRAMBE le coorti (generation 1 e 2,
secondo db.get_significant_results()), il genotipo binario (0 = non mutato,
1 = mutato) per ciascun paziente, producendo due CSV (uno per coorte)
pazienti x varianti.

Versione ottimizzata per file "full genome" con centinaia di migliaia di colonne:
  - Pre-filtro colonne con awk (stream, C-level split) prima di passare a pandas,
    invece di lasciare che il parser C di pandas tokenizzi ogni riga per intero
    con usecols (che scarta le colonne DOPO averle comunque splittate).
  - Binarizzazione vettorizzata con numpy invece di applymap cella-per-cella.
  - Parallelizzazione per RIGHE (pazienti), non per varianti: splittare per
    variante non riduce il lavoro (ogni riga va comunque interamente
    tokenizzata per estrarre anche una sola colonna), quindi il file viene
    diviso in N chunk di righe e l'estrazione delle 34 colonne gira in
    parallelo su ciascun chunk, poi i risultati vengono concatenati.

Fonte varianti significative : db.get_significant_results()
Fonte genotipi               : i file genetici RAW per generazione (stesso formato
                                usato da data_loader.py: colonna id/IID + una colonna
                                per variante, nome "{chrom}_{pos}_{mutation}",
                                valori dosaggio 0/1/2/-1/. ecc.)

Usage: python extract_significant_variant_matrices.py
"""
import os
import glob
import time
import logging
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from db import get_significant_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────
GENOTYPE_FILES = {
    1: "/mnt/cresla_prod/genome_datasets/merged_csv/full_chr_gen1_test1.csv",
    2: "/mnt/cresla_prod/genome_datasets/merged_csv/gen2_variants.csv",
}
OUT_DIR = "/srv/python-projects/gene-environment/significant_variant_matrices"
TMP_DIR = "/tmp/significant_variant_extract"
N_WORKERS = os.cpu_count() or 4  # numero di chunk di righe in cui dividere ciascun file
# ─────────────────────────────────────────────────────────────────────────────


def build_variant_label(chromosome, position, mutation):
    return f"{chromosome}_{position}_{mutation}"


def load_id_col(header):
    """Trova la colonna id, gestendo il rename IID -> id fatto da data_loader.py."""
    if "id" in header:
        return "id"
    if "IID" in header:
        return "IID"
    raise ValueError(f"Nessuna colonna id/IID trovata nell'header: {header[:10]}...")


def binarize_vectorized(df_geno):
    """
    Converte dosaggio in presenza mutazione binaria (vettorizzato, no loop Python):
      0            -> 0 (non mutato)
      1 oppure 2   -> 1 (mutato)
      qualsiasi altro valore (-1, ., NaN, ...) -> NA (missing, escluso a valle)
    """
    df_num = df_geno.apply(pd.to_numeric, errors="coerce")
    arr = np.select(
        [df_num.eq(0), df_num.isin([1, 2])],
        [0, 1],
        default=np.nan,
    )
    return pd.DataFrame(arr, index=df_geno.index, columns=df_geno.columns)


def _awk_extract_chunk(args):
    """
    Estrae le colonne richieste da UN chunk di righe (worker per ProcessPoolExecutor).
    Girato in parallelo su tutti i chunk: il parallelismo e' sulle RIGHE (pazienti),
    non sulle varianti, perche' il costo di tokenizzare una riga e' quasi tutto
    fisso indipendentemente da quante colonne vengono tenute alla fine -> frazionare
    per variante rilegge l'intero file N volte invece di dividerne il costo per N.
    """
    chunk_path, awk_field_list, out_path = args
    awk_cmd = f"awk -F',' 'BEGIN{{OFS=\",\"}} {{print {awk_field_list}}}' {chunk_path} > {out_path}"
    subprocess.run(awk_cmd, shell=True, check=True)
    return out_path


def extract_columns_parallel(path, id_col_pos, wanted_positions, tmp_out, tmp_dir,
                              n_workers=N_WORKERS, has_header=True):
    """
    Pre-filtra le colonne a livello di stream con awk, PRIMA di passare a pandas,
    parallelizzando sui BLOCCHI DI RIGHE del file (non sulle varianti):

      1. split del file dati (header escluso) in n_workers chunk, preservando
         i confini di riga (split -n l/N non spezza mai una riga a meta').
      2. ogni chunk viene processato da un worker awk indipendente che estrae
         id + le colonne richieste (stesso comando per tutti, cambia solo il
         file di input).
      3. i risultati vengono concatenati in ordine nel file finale.

    Questo distribuisce il costo reale (tokenizzare ogni riga del file
    full-genome) su piu' core, mentre il numero di varianti resta ininfluente
    sul grado di parallelismo.
    """
    all_positions = [id_col_pos] + wanted_positions
    awk_field_list = ",".join(f"${p}" for p in all_positions)

    chunk_prefix = os.path.join(tmp_dir, os.path.basename(path) + ".chunk_")

    t0 = time.perf_counter()
    skip = "tail -n +2" if has_header else "cat"
    split_cmd = f"{skip} {path} | split -n l/{n_workers} -d -a 3 - {chunk_prefix}"
    subprocess.run(split_cmd, shell=True, check=True)
    log.info("  split in chunk completato in %.1fs", time.perf_counter() - t0)

    chunk_files = sorted(glob.glob(chunk_prefix + "*"))
    if not chunk_files:
        chunk_files = [path]
    n_chunks = len(chunk_files)
    log.info("  avvio estrazione colonne su %d chunk (n_workers=%d)", n_chunks, n_workers)

    tasks = [(cf, awk_field_list, f"{cf}.out") for cf in chunk_files]

    t0 = time.perf_counter()
    out_files_by_task = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_awk_extract_chunk, task): task for task in tasks}
        done_count = 0
        for fut in as_completed(futures):
            task = futures[fut]
            out_files_by_task[task] = fut.result()
            done_count += 1
            log.info("  chunk completato: %d/%d", done_count, n_chunks)
    log.info("  estrazione parallela completata in %.1fs", time.perf_counter() - t0)

    # ordine originale dei chunk (as_completed non garantisce l'ordine di arrivo)
    out_files = [out_files_by_task[task] for task in tasks]

    t0 = time.perf_counter()
    with open(tmp_out, "w") as fout:
        for of in out_files:
            with open(of, "r") as fin:
                fout.write(fin.read())
    log.info("  concatenazione chunk completata in %.1fs", time.perf_counter() - t0)

    for cf in chunk_files:
        if cf != path and os.path.exists(cf):
            os.remove(cf)
    for of in out_files:
        if os.path.exists(of):
            os.remove(of)


def extract_cohort(generation, variant_labels, out_dir, tmp_dir):
    t_start = time.perf_counter()
    log.info("=== Coorte %d: inizio ===", generation)

    path = GENOTYPE_FILES[generation]
    if not os.path.exists(path):
        log.warning("File non trovato per generation %d: %s", generation, path)
        return f"generation {generation}: file non trovato"

    t0 = time.perf_counter()
    header = pd.read_csv(path, nrows=0).columns.tolist()
    log.info("  header letto (%d colonne totali) in %.1fs", len(header), time.perf_counter() - t0)

    id_col = load_id_col(header)
    id_col_pos = header.index(id_col) + 1  # awk è 1-based

    present = [v for v in variant_labels if v in header]
    missing = [v for v in variant_labels if v not in header]
    if missing:
        log.warning("Coorte %d: %d varianti non trovate nel file:", generation, len(missing))
        for m in missing:
            log.warning("     - %s", m)

    if not present:
        log.error("Coorte %d: nessuna variante significativa presente nel file.", generation)
        return f"generation {generation}: nessuna variante trovata"

    log.info("  %d/%d varianti presenti nel file per questa coorte", len(present), len(variant_labels))

    wanted_positions = [header.index(v) + 1 for v in present]

    os.makedirs(tmp_dir, exist_ok=True)
    tmp_out = os.path.join(tmp_dir, f"gen{generation}_filtered.csv")

    extract_columns_parallel(path, id_col_pos, wanted_positions, tmp_out, tmp_dir)

    t0 = time.perf_counter()
    filtered_col_names = ["id"] + present
    df = pd.read_csv(tmp_out, header=None, names=filtered_col_names, dtype=str)
    df = df.set_index("id")
    log.info("  file filtrato letto in pandas (%d pazienti) in %.1fs", len(df), time.perf_counter() - t0)

    t0 = time.perf_counter()
    df_bin = binarize_vectorized(df)
    log.info("  binarizzazione completata in %.1fs", time.perf_counter() - t0)

    out_path = os.path.join(out_dir, f"cohort{generation}_significant_variants.csv")
    df_bin.to_csv(out_path)

    os.remove(tmp_out)

    msg = f"Coorte {generation}: {out_path}  ({len(df_bin)} pazienti, {df_bin.shape[1]} varianti)"
    log.info("  -> %s", msg)
    log.info("=== Coorte %d: completata in %.1fs totali ===", generation, time.perf_counter() - t_start)
    return msg


def main():
    log.info("=== Recupero varianti significative (entrambe le coorti) dal DB ===")
    t_total = time.perf_counter()
    sig = get_significant_results()
    if sig.empty:
        log.info("Nessuna variante significativa trovata in entrambe le coorti. Esco.")
        return

    sig["label"] = sig.apply(
        lambda r: build_variant_label(r["chromosome"], r["position"], r["mutation"]), axis=1
    )
    unique_labels = sorted(sig["label"].unique())
    log.info("%d varianti uniche da estrarre.", len(unique_labels))

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Le due coorti vengono processate in sequenza; il parallelismo (N_WORKERS
    # processi awk) e' interno a ciascuna coorte, sui chunk di righe del file,
    # per evitare di dividere gli stessi core della macchina su 2 pool
    # concorrenti da N_WORKERS ciascuno (2 x N_WORKERS processi contemporanei
    # saturerebbero comunque la CPU/il disco senza guadagno aggiuntivo).
    for gen in (1, 2):
        try:
            extract_cohort(gen, unique_labels, OUT_DIR, TMP_DIR)
        except Exception as e:
            log.error("Coorte %d fallita: %s", gen, e)

    log.info("=== Fatto in %.1fs totali ===", time.perf_counter() - t_total)


if __name__ == "__main__":
    main()