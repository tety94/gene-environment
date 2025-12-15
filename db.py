# db.py
import mysql.connector
from mysql.connector import pooling
from config import DB_USER, DB_PASSWORD, DB_NAME
import pandas as pd


def get_conn():
    """Crea una connessione nuova (ogni processo ne avrà una)"""
    conn = mysql.connector.connect(
        host="localhost",
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        autocommit=True
    )
    return conn

def gene_already_done(conn, gene):
    cur = conn.cursor()
    cur.execute("SELECT gene FROM gene_results WHERE gene=%s AND completed=1", (gene,))
    r = cur.fetchone()
    cur.close()
    return r is not None


def save_gene_result(conn, gene, mutati, non_mutati, obs_coef, mean_coef, sd_coef, empirical_p):
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO gene_results (gene, mutati, non_mutati, obs_coef, mean_coef, sd_coef, empirical_p, completed)
            VALUES (%s,%s,%s,%s,%s,%s,%s,1)
            ON DUPLICATE KEY UPDATE 
                mutati=VALUES(mutati),
                non_mutati=VALUES(non_mutati),
                obs_coef=VALUES(obs_coef),
                mean_coef=VALUES(mean_coef),
                sd_coef=VALUES(sd_coef),
                empirical_p=VALUES(empirical_p),
                completed=1
        """, (gene, mutati, non_mutati, obs_coef, mean_coef, sd_coef, empirical_p))
    finally:
        cur.close()

def load_gene_results():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT gene, obs_coef, empirical_p FROM gene_results WHERE completed=1")
        rows = cur.fetchall()
        # costruisci DataFrame manualmente
        df = pd.DataFrame(rows, columns=["gene", "obs_coef", "empirical_p"])
    finally:
        cur.close()
        conn.close()
    return df
