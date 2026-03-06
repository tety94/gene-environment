from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from data_loader import load_and_prepare_data
from modeling import process_single_variant
from utils import add_fdr, volcano_plot
from config import MAX_WORKERS
from db import load_variant_results, insert_new_variants, get_variants_to_run, save_variant_result
import random
import pickle
import os

BATCH_SIZE = 100  # blocco di insert nel DB

global_df = None

def init_worker():
    import modeling
    import db

    # Carica dataframe globale
    with open("temp_df.pkl", "rb") as f:
        modeling.global_df = pickle.load(f)

    # Connessione DB unica per worker
    modeling.worker_conn = db.get_conn()
    print(f"[INFO] Worker {os.getpid()} caricato global_df e connessione DB")

def run_parallel_processing(variants, mapping, Ecols, description=""):
    print(f"[INFO] Avvio processi paralleli: {description} ({len(variants)} varianti)")

    buffer = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as ex:
        futures = [ex.submit(process_single_variant, g, mapping[g], Ecols) for g in variants]

        for f in as_completed(futures):
            try:
                res = f.result()
                if res:
                    buffer.append(res)

                # Inserisci a blocchi
                if len(buffer) >= BATCH_SIZE:
                    save_variant_result(buffer)
                    buffer = []

            except Exception as e:
                print("Errore in un processo:", e)

        # Inserisci eventuale residuo
        if buffer:
            save_variant_result(buffer)

def main():
    start_time = datetime.now()
    print(f"[START] Analisi iniziata alle: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    df, variant_cols_safe, mapping, Ecols, variant_cols = load_and_prepare_data()
    df.to_pickle("temp_df.pkl")

    # Inserimento varianti nuove nel DB
    variants_to_insert = []
    for v in variant_cols:
        parts = v.split("_", 2)
        variants_to_insert.append({
            "variant": v,
            "chromosome": parts[0] if len(parts) > 0 else None,
            "position": int(parts[1]) if len(parts) > 1 else None,
            "mutation": parts[2] if len(parts) > 2 else None
        })
    insert_new_variants(variants_to_insert)

    variants_to_run = get_variants_to_run(mapping, variant_cols_safe)
    random.shuffle(variants_to_run)

    run_parallel_processing(variants_to_run, mapping, Ecols, description="primo run con permutazioni standard")

    # ---------- CARICA RISULTATI E PLOT ----------
    results_df = load_variant_results()
    results_df = add_fdr(results_df)
    volcano_plot(results_df, save_path="volcano_plot_final.png")
    print(f"[INFO] Test permutazioni completato. Volcano plot salvato.")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"[END] Analisi terminata alle: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DURATION] Tempo totale: {str(duration)}")

    # Chiudi connessione di ogni worker (globalmente importando modeling)
    import modeling
    modeling.worker_conn.close()

if __name__ == "__main__":
    main()