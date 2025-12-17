from concurrent.futures import ProcessPoolExecutor, as_completed
from pipeline.gene_annotator import GeneAnnotator
from db import get_genes_to_annotate
from config import MAX_WORKERS
from tqdm import tqdm

def run_parallel_processing(genes):
    print(f"[INFO] Avvio processi paralleli: ({len(genes)} geni)")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(GeneAnnotator.annotate, g): g for g in genes}

        for f in tqdm(as_completed(futures), total=len(futures), desc="Annotating genes"):
            try:
                f.result()
            except Exception as e:
                print(f"Errore sul gene {futures[f]}:", e)

if __name__ == "__main__":
    genes = get_genes_to_annotate()
    run_parallel_processing(genes)
