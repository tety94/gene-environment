# db.py
import mysql.connector
from mysql.connector import pooling
from config import DB_USER, DB_PASSWORD, DB_NAME, DB_HOST, DB_PORT
import pandas as pd
import math
import numpy as np

def get_conn():
    """Crea una connessione nuova (ogni processo ne avrà una)"""
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
        cur.execute("SELECT variant, mutation, position, chromosome FROM variant_results WHERE gene IS NULL AND empirical_p < 0.05")
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

def save_variant_result(conn, variant, mutati, non_mutati, obs_coef, mean_coef, sd_coef, empirical_p, iterations, balance):
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
                sd_coef, empirical_p, iterations, balance, completed 
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1)
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
                completed=1
        """, (
            variant, gene, chromosome, position, mutation, mutati, non_mutati,
            safe_val(obs_coef),
            safe_val(mean_coef),
            safe_val(sd_coef),
            safe_val(empirical_p),
            iterations,
            safe_val(balance)
        ))
    finally:
        cur.close()

def load_variant_results():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT variant, obs_coef, empirical_p FROM variant_results WHERE completed=1")
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
            (variant, chromosome, position, mutation, completed, in_progress)
        VALUES (%s,%s,%s,%s,0,0)
        ON DUPLICATE KEY UPDATE variant=variant
        """
        data = []
        for v in variants:
            chrom = v.get("chromosome")
            # se chromosome è un numero valido lo converte a int, altrimenti lascia None
            chrom = int(chrom) if chrom is not None and str(chrom).isdigit() else None
            pos = v.get("position")
            pos = int(pos) if pos is not None and str(pos).isdigit() else None
            data.append((v["variant"], chrom, pos, v.get("mutation")))

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
    # Connessione al DB
    conn = get_conn()
    cur = conn.cursor()

    # Recupera tutte le varianti già completate o in progress
    cur.execute("""
        SELECT variant 
        FROM variant_results 
        WHERE completed=1 OR in_progress=1
    """)
    done_variants = set(row[0] for row in cur.fetchall())

    cur.close()
    conn.close()

    # Filtra le varianti da processare
    variants_to_run = [v_safe for v_safe in variant_cols_safe if mapping[v_safe] not in done_variants]

    print(f"[INFO] Varianti da processare: {len(variants_to_run)}")
    return variants_to_run
