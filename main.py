from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from data_loader import load_and_prepare_data
from modeling import process_single_variant
from utils import add_fdr, volcano_plot
from config import MAX_WORKERS, PVALUE_THRESHOLD, N_PERM_HIGH
from db import load_variant_results, get_conn, delete_variants, insert_new_variants, get_variants_to_run
import config
import random

#todo: valutare split in discovery replication ed eventualmente valutare solo il beta concorde
#todo: selezionare le varienti in base al linkage disequilibrium

def run_parallel_processing(variants, mapping, Ecols, description=""):
    print(f"[INFO] Avvio processi paralleli: {description} ({len(variants)} varianti)")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_single_variant, g, mapping[g], Ecols) for g in variants]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("Errore in un processo:", e)

def main():
    start_time = datetime.now()
    print(f"[START] Analisi iniziata alle: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    df, variant_cols_safe, mapping, Ecols, variant_cols = load_and_prepare_data()
    df.to_pickle("temp_df.pkl")

    # Prepara la lista di dizionari da inserire nel DB
    variants_to_insert = []
    for v in variant_cols:
        parts = v.split("_", 2)
        chromosome = parts[0] if len(parts) > 0 else None
        position = int(parts[1]) if len(parts) > 1 else None
        mutation = parts[2] if len(parts) > 2 else None

        variants_to_insert.append({
            "variant": v,
            "chromosome": chromosome,
            "position": position,
            "mutation": mutation
        })

    # Inserisci nel DB
    insert_new_variants(variants_to_insert)

    # ---------- PRIMO RUN ----------
    variants_to_run = get_variants_to_run(mapping, variant_cols_safe)
    random.shuffle(variants_to_run)

    run_parallel_processing(variants_to_run, mapping, Ecols, description="primo run con permutazioni standard")

    # ---------- CARICA RISULTATI E PLOT ----------
    results_df = load_variant_results()
    results_df = add_fdr(results_df)
    volcano_plot(results_df, save_path="volcano_plot.png")

    # ---------- FILTRO RISULTATI SIGNIFICATIVI ----------
    sig_variants_df = results_df[results_df['empirical_p'] < PVALUE_THRESHOLD]
    sig_variants = sig_variants_df['variant'].tolist()
    print(f"[INFO] Numero di geni significativi (p<{PVALUE_THRESHOLD}): {len(sig_variants)}")

    if sig_variants:
        # cancella dal DB i geni significativi
        conn = get_conn()
        delete_variants(conn, sig_variants)
        conn.close()

        # Aggiorna temporaneamente il numero di permutazioni
        config.N_PERM = N_PERM_HIGH
        random.shuffle(sig_variants)
        run_parallel_processing(sig_variants, mapping, Ecols, description=f"secondo run con {N_PERM_HIGH} permutazioni")

        # ricarica risultati finali
        results_df = load_variant_results()
        results_df = add_fdr(results_df)
        volcano_plot(results_df, save_path="volcano_plot_final.png")
        print(f"[INFO] Test permutazioni avanzate completato. Volcano plot finale salvato.")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"[END] Analisi terminata alle: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DURATION] Tempo totale impiegato: {str(duration)}")

if __name__ == "__main__":
    main()