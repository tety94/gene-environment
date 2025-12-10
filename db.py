# db.py
import mysql.connector
from mysql.connector import pooling
from config import DB_USER, DB_PASSWORD

pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=10,
    host="localhost",
    user=DB_USER,
    password=DB_PASSWORD,
    database="cresla_definitivo"
)

def get_conn():
    return pool.get_connection()

def gene_already_done(gene):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT gene FROM gene_results WHERE gene=%s AND completed=1", (gene,))
    r = cur.fetchone()
    cur.close()
    conn.close()
    return r is not None

def save_gene_iteration(gene, iteration, coef, pvalue, n):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO gene_iterations (gene, iteration, coef, pvalue, n)
            VALUES (%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE coef=VALUES(coef), pvalue=VALUES(pvalue), n=VALUES(n)
        """, (gene, iteration, coef, pvalue, n))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def save_gene_result(gene, mutati, non_mutati, obs_coef, mean_coef, sd_coef, empirical_p):
    conn = get_conn()
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
        conn.commit()
    finally:
        cur.close()
        conn.close()
