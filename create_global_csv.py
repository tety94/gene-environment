import os
import pandas as pd
from glob import glob
from config import VFC_FOLDERS, NULL_PRECENTAGE
from concurrent.futures import ProcessPoolExecutor, as_completed

input_folders = VFC_FOLDERS
output_folder = "/mnt/cresla_prod/genome_datasets/merged_csv"
os.makedirs(output_folder, exist_ok=True)

chromosomes = [str(i) for i in range(1, 23)]

def merge_chromosome(chr_num):
    print(f"\n🔹 Processing chromosome {chr_num}")

    # Trova tutti i CSV per questo cromosoma
    csv_files = []
    for folder in input_folders:
        pattern = os.path.join(folder, f"vcf_filtered/genotypes_matrix/*chr{chr_num}.vcf_filtered_genotypes.csv")
        csv_files.extend(glob(pattern))

    if not csv_files:
        print(f"⚠️ Nessun CSV trovato per chr{chr_num}")
        return

    dfs = []
    for csv_file in csv_files:
        print(f"  Leggo {csv_file}")
        df = pd.read_csv(csv_file, index_col=0)
        dfs.append(df)

    # Concatenate verticalmente (righe dei campioni)
    merged_df = pd.concat(dfs, axis=0, join="outer")

    # Riempi NaN con -1
    merged_df = merged_df.fillna(-1).astype(int)

    # Filtra colonne con troppi valori mancanti
    merged_df = merged_df.loc[:, (merged_df == -1).mean() < NULL_PRECENTAGE]

    # Rimuovi eventuali duplicati di campioni
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

    # Salva CSV finale
    output_csv = os.path.join(output_folder, f"chr{chr_num}_merged.csv")
    merged_df.to_csv(output_csv)
    print(f"✅ CSV unito salvato in: {output_csv}")

# -------------------------------
# Parallelizzazione
# -------------------------------
with ProcessPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(merge_chromosome, chr_num) for chr_num in chromosomes]
    for future in as_completed(futures):
        future.result()
