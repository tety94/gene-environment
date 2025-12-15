from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from data_loader import load_and_prepare_data
from modeling import process_single_gene
from utils import add_fdr, volcano_plot
from config import MAX_WORKERS
from db import load_gene_results

#TODO: non fare montecarlo se beta osservato è vicino allo 0
#todo: permutazioni 500 per poi eventualmetnte farne 10000
#todo: valutare split in discovery replication ed eventualmente valutare solo il beta concorde
#todo: selezionare le varienti in base al linkage disequilibrium

def main():
    start_time = datetime.now()
    print(f"[START] Analisi iniziata alle: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    df, gene_cols_safe, mapping, Ecols = load_and_prepare_data()
    df.to_pickle("temp_df.pkl")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_single_gene, gc, mapping[gc], Ecols) for gc in gene_cols_safe]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("Errore in un processo:", e)

    results_df = load_gene_results()
    results_df = add_fdr(results_df)
    volcano_plot(results_df, save_path="volcano_plot.png")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"[END] Analisi terminata alle: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DURATION] Tempo totale impiegato: {str(duration)}")

if __name__ == "__main__":
    main()