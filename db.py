# db.py
import mysql.connector
from mysql.connector import pooling
from config import DB_USER, DB_PASSWORD, DB_NAME, DB_HOST, DB_PORT
import pandas as pd


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
    cur.execute("SELECT variant FROM variant_results WHERE variant=%s AND completed=1", (variant,))
    r = cur.fetchone()
    cur.close()
    return r is not None

def get_empty_variants_gene():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT variant, mutation, position, chromosome FROM variant_results WHERE gene IS NULL AND empirical_p < 0.05")
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=["variant", "mutatio", "position", "chromosome"])
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
            variant, gene, chromosome, position, mutation, mutati, non_mutati, obs_coef, mean_coef,
            sd_coef, empirical_p, iterations, balance
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
        data = [
            (v["variant"], v.get("chromosome"), v.get("position"), v.get("mutation"), 0, 0)
            for v in variants
        ]
        cursor.executemany(sql, data)
        conn.commit()
        print(f"[INFO] Inserite/aggiornate {cursor.rowcount} varianti nel DB")
    finally:
        cursor.close()
        conn.close()

    return cursor.rowcount