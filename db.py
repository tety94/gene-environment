# db.py
import mysql.connector
from config import DB_USER, DB_PASSWORD, DB_NAME, DB_HOST, DB_PORT, EXPOSURE, N_PERM_HIGH, PVALUE_THRESHOLD, N_PERM
import pandas as pd
import math
import numpy as np


def get_conn():
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        autocommit=True
    )
    return conn


def variant_already_done(conn, variant):
    cur = conn.cursor()
    cur.execute("""
        SELECT completed, in_progress 
        FROM variant_results 
        WHERE variant=%s
    """, (variant,))
    r = cur.fetchone()
    cur.close()

    if r is None:
        return False  # variante non presente → da lanciare
    completed, in_progress = r
    return completed or in_progress  # se completata o in elaborazione → NON lanciare


def get_empty_variants_gene():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT variant, mutation, position, chromosome FROM variant_results "
                    " WHERE gene IS NULL AND empirical_p < %s AND iterations = %s",
                    (PVALUE_THRESHOLD, N_PERM_HIGH))
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=["variant", "mutation", "position", "chromosome"])
    finally:
        cur.close()
        conn.close()
    return df


def update_variant_gene(conn, variant, gene_id, gene_name):
    cur = conn.cursor()
    cur.execute("""
        UPDATE variant_results
        SET gene=%s, gene_name=%s
        WHERE variant=%s AND gene IS NULL
    """, (gene_id, gene_name, variant))
    conn.commit()
    cur.close()


def save_variant_result(conn, variant, mutati, non_mutati, obs_coef, mean_coef, sd_coef, empirical_p, iterations,
                        balance):
    cur = conn.cursor()

    parts = variant.split("_", 2)  # split massimo 2, così l'ultima parte resta tutta la mutazione
    chromosome = parts[0] if len(parts) > 0 else None
    position = int(parts[1]) if len(parts) > 1 else None
    mutation = parts[2] if len(parts) > 2 else None
    gene = None  # non presente in questo formato

    try:
        cur.execute("""
            INSERT INTO variant_results (
                variant, gene, chromosome, position, mutation, mutati, non_mutati, obs_coef, mean_coef, 
                sd_coef, empirical_p, iterations, balance, exposure, completed 
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1)
            ON DUPLICATE KEY UPDATE 
                gene=VALUES(gene),
                chromosome=VALUES(chromosome),
                position=VALUES(position),
                mutation=VALUES(mutation),
                mutati=VALUES(mutati),
                non_mutati=VALUES(non_mutati),
                obs_coef=VALUES(obs_coef),
                mean_coef=VALUES(mean_coef),
                sd_coef=VALUES(sd_coef),
                empirical_p=VALUES(empirical_p),
                iterations=VALUES(iterations),
                balance=VALUES(balance),
                exposure=VALUES(exposure),
                completed=1
        """, (
            variant, gene, chromosome, position, mutation, mutati, non_mutati,
            safe_val(obs_coef),
            safe_val(mean_coef),
            safe_val(sd_coef),
            safe_val(empirical_p),
            safe_val(iterations),
            safe_val(balance),
            EXPOSURE
        ))
    finally:
        cur.close()


def load_variant_results():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT variant, obs_coef, empirical_p FROM variant_results WHERE exposure=%s AND completed=1 AND iterations =%s",
            (EXPOSURE, N_PERM_HIGH))
        rows = cur.fetchall()
        # costruisci DataFrame manualmente
        df = pd.DataFrame(rows, columns=["variant", "obs_coef", "empirical_p"])
    finally:
        cur.close()
        conn.close()
    return df


def delete_variants(conn, variant_list):
    if not variant_list:
        return
    cursor = conn.cursor()
    format_strings = ','.join(['%s'] * len(variant_list))
    cursor.execute(f"DELETE FROM variant_results WHERE variant IN ({format_strings})", tuple(variant_list))
    conn.commit()
    cursor.close()


def insert_new_variants(variants):
    """
    Inserisce nuove varianti nella tabella variant_results.

    Args:
        variants (list of dict): lista di dizionari, ognuno con chiavi:
            - variant
            - chromosome
            - position
            - mutation
    """
    if not variants:
        return 0

    conn = get_conn()
    cursor = conn.cursor()

    try:
        sql = """
        INSERT INTO variant_results 
            (variant, chromosome, position, mutation, exposure, iterations)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE variant=variant
        """
        data = []
        for v in variants:
            chrom = v.get("chromosome")
            # se chromosome è un numero valido lo converte a int, altrimenti lascia None
            chrom = int(chrom) if chrom is not None and str(chrom).isdigit() else None
            pos = v.get("position")
            pos = int(pos) if pos is not None and str(pos).isdigit() else None
            data.append((v["variant"], chrom, pos, v.get("mutation"), EXPOSURE, N_PERM))

        cursor.executemany(sql, data)
        conn.commit()
        print(f"[INFO] Inserite/aggiornate {cursor.rowcount} varianti nel DB")
    finally:
        cursor.close()
        conn.close()

    return cursor.rowcount


def mark_variant_in_progress(conn, variant):
    """
    Segna una variante come in progress.
    Ritorna True se è stata aggiornata (quindi può essere elaborata), False se qualcun altro l'ha già presa.
    """
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE variant_results
            SET in_progress=1
            WHERE variant=%s AND completed=0 AND in_progress=0
        """, (variant,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        cur.close()


def reset_variant_in_progress(conn, variant, success=True):
    """
    Resetta lo stato di in_progress.
    Se success=True, imposta anche completed=1.
    """
    cur = conn.cursor()
    try:
        if success:
            cur.execute("""
                UPDATE variant_results
                SET completed=1, in_progress=0
                WHERE variant=%s
            """, (variant,))
        else:
            cur.execute("""
                UPDATE variant_results
                SET in_progress=0
                WHERE variant=%s
            """, (variant,))
        conn.commit()
    finally:
        cur.close()


def safe_val(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, (np.float32, np.float64)) and np.isnan(x):
        return None
    return x


def get_variants_to_run(mapping, variant_cols_safe):
    """
    Restituisce le varianti safe da processare, escludendo quelle già completate o in progress.
    mapping: dict safe -> originale
    variant_cols_safe: lista delle colonne safe del DF
    """
    # costruisci mapping inverso: originale -> safe
    orig_to_safe = {v: k for k, v in mapping.items()}

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT variant 
        FROM variant_results 
        WHERE (completed=1 OR in_progress=1) 
          AND exposure=%s 
          AND iterations=%s
    """, (EXPOSURE, N_PERM))

    done_variants = set(row[0] for row in cur.fetchall())
    print(f"[INFO] done_variants dal DB: {len(done_variants)}")

    cur.close()
    conn.close()

    # Converti i nomi originali in safe
    done_variants_safe = {orig_to_safe[v] for v in done_variants if v in orig_to_safe}

    # Filtra le varianti da processare
    variants_to_run = [v for v in variant_cols_safe if v not in done_variants_safe]

    print(f"[INFO] Varianti da processare: {len(variants_to_run)}")
    return variants_to_run


def get_genes_to_annotate():
    """
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
SELECT distinct vr.gene
FROM variant_results_significant vr 
LEFT JOIN gene_neuro_annotation gna 
	ON vr.gene = gna.gene_id 
where vr.gene is not null and vr.gene != 'NO-GENE'
and gna.gene_id IS NULL     
    """)

    genes = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()

    return genes


# GESTIONE GENI
def upsert_gene_neuro_annotation(data: dict):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    sql = """
    INSERT INTO gene_neuro_annotation (
        gene_id,
        gene_symbol,
        gene_type,
        expressed_brain,
        brain_tissues,
        expressed_neurons,
        expressed_glia,
        cell_types,
        go_neuro_processes,
        go_toxic_response,
        ctd_chemicals,
        ctd_neuro_diseases,
        neuro_plausibility_score
    ) VALUES (
        %(gene_id)s,
        %(gene_symbol)s,
        %(gene_type)s,
        %(expressed_brain)s,
        %(brain_tissues)s,
        %(expressed_neurons)s,
        %(expressed_glia)s,
        %(cell_types)s,
        %(go_neuro_processes)s,
        %(go_toxic_response)s,
        %(ctd_chemicals)s,
        %(ctd_neuro_diseases)s,
        %(neuro_plausibility_score)s
    )
    ON DUPLICATE KEY UPDATE
        gene_symbol = VALUES(gene_symbol),
        gene_type = VALUES(gene_type),
        expressed_brain = VALUES(expressed_brain),
        brain_tissues = VALUES(brain_tissues),
        expressed_neurons = VALUES(expressed_neurons),
        expressed_glia = VALUES(expressed_glia),
        cell_types = VALUES(cell_types),
        go_neuro_processes = VALUES(go_neuro_processes),
        go_toxic_response = VALUES(go_toxic_response),
        ctd_chemicals = VALUES(ctd_chemicals),
        ctd_neuro_diseases = VALUES(ctd_neuro_diseases),
        neuro_plausibility_score = VALUES(neuro_plausibility_score),
        last_updated = CURRENT_TIMESTAMP
    """

    cur.execute(sql, data)
    cur.close()
    conn.close()


def get_gene_neuro_annotation(gene_id: str):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    sql = """
    SELECT *
    FROM gene_neuro_annotation
    WHERE gene_id = %s
    """

    cur.execute(sql, (gene_id,))
    result = cur.fetchone()

    cur.close()
    conn.close()
    return result
