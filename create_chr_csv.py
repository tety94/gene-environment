import os
import pandas as pd
import numpy as np
from glob import glob
from config import VFC_FOLDERS, NULL_PRECENTAGE, OUTPUT_FOLDER
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------------
# Configurazione
# -------------------------------
input_folders = VFC_FOLDERS        # Cartelle contenenti CSV dei genotipi dei campioni
output_folder = OUTPUT_FOLDER       # Cartella dove salvare i CSV uniti per cromosoma
os.makedirs(output_folder, exist_ok=True)

chromosomes = [str(i) for i in range(1, 23)]  # Cromosomi da processare

# -------------------------------
# Funzione principale
# -------------------------------
def merge_chromosome(chr_num):
    """
    Unisce tutti i CSV di genotipi per un cromosoma,
    riempie valori mancanti, filtra SNP con troppe missing values,
    binarizza i genotipi e salva un CSV finale pronto per analisi genetiche.
    """
    print(f"\n🔹 Processing chromosome {chr_num}")

    # Trova tutti i CSV relativi a questo cromosoma nelle cartelle di input
    csv_files = []
    for folder in input_folders:
        pattern = os.path.join(folder, f"vcf_filtered/genotypes_matrix/*chr{chr_num}.vcf_filtered_genotypes.csv")
        csv_files.extend(glob(pattern))

    if not csv_files:
        print(f"⚠️ Nessun CSV trovato per chr{chr_num}")
        return

    # Legge tutti i CSV trovati e li mette in una lista di DataFrame
    dfs = []
    for csv_file in csv_files:
        print(f"  Leggo {csv_file}")
        df = pd.read_csv(csv_file, index_col=0)  # righe = campioni, colonne = SNP
        dfs.append(df)

    # ---------------- UNIONE DEI DATI ----------------
    # Concatenazione verticale: aggiunge nuovi campioni (righe) mantenendo tutte le varianti (colonne)
    merged_df = pd.concat(dfs, axis=0, join="outer")

    # ---------------- GESTIONE DEI MISSING ----------------
    # Valori mancanti (-1) per campioni che non hanno un SNP specifico
    merged_df = merged_df.fillna(-1).astype(int)

    # Filtra le colonne (SNP) che hanno troppi valori mancanti
    # Esempio: se NULL_PRECENTAGE=0.1, rimuove SNP con >=10% genotipi mancanti
    merged_df = merged_df.loc[:, (merged_df == -1).mean() < NULL_PRECENTAGE]

    # ---------------- BINARIZZAZIONE DEI GENOTIPI ----------------
    # Trasforma i genotipi numerici (0,1,2) in binari:
    # 0 → assenza di allele alternativo
    # 1 o 2 → presenza di almeno un allele alternativo
    variant_cols = merged_df.columns
    arr = merged_df.values

    #test 1
    # arr[arr < 0] = 0   # valori mancanti diventano 0
    # arr[arr > 0] = 1   # 1 o 2 diventano 1

    #test 2
    arr[arr < 0] = 0   # valori mancanti diventano 0 mentre 1 e 2 restano così

    merged_df[:] = arr.astype(np.int8)

    # ---------------- RIMOZIONE DUPLICATI ----------------
    # Se ci sono campioni ripetuti, mantiene solo la prima occorrenza
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

    # ---------------- SALVATAGGIO ----------------
    output_csv = os.path.join(output_folder, f"chr{chr_num}_merged.csv")
    merged_df.to_csv(output_csv)
    print(f"✅ CSV unito e binarizzato salvato in: {output_csv}")

# -------------------------------
# Parallelizzazione per cromosoma
# -------------------------------
with ProcessPoolExecutor(max_workers=16) as executor:
    # Submit delle funzioni in parallelo per ogni cromosoma
    futures = [executor.submit(merge_chromosome, chr_num) for chr_num in chromosomes]
    for future in as_completed(futures):
        future.result()  # attende che ogni cromosoma finisca

