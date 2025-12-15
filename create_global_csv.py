import os
import pandas as pd
from glob import glob
from config import VFC_FOLDERS, NULL_PRECENTAGE

# -------------------------------
# Configurazione
# -------------------------------
input_folders = VFC_FOLDERS
output_folder = "/mnt/cresla_prod/genome_datasets/merged_csv"
os.makedirs(output_folder, exist_ok=True)

# Cromosomi da processare
chromosomes = [str(i) for i in range(1, 23)]

for chr_num in chromosomes:
    print(f"\n🔹 Processing chromosome {chr_num}")

    # Trova tutti i CSV per questo cromosoma
    csv_files = []
    for folder in input_folders:
        pattern = os.path.join(folder, f"vcf_filtered/genotypes_matrix/*chr{chr_num}_genotypes.csv")
        csv_files.extend(glob(pattern))

    if not csv_files:
        print(f"⚠️ Nessun CSV trovato per chr{chr_num}")
        continue

    merged_df = pd.DataFrame()

    # Leggi e concatena
    for csv_file in csv_files:
        print(f"  Leggo {csv_file}")
        df = pd.read_csv(csv_file, index_col=0)
        merged_df = pd.concat([merged_df, df], axis=1)

    # Rimuovi eventuali colonne duplicate (stesso variant_id in più file)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Sostituisci NaN con -1
    merged_df = merged_df.fillna(-1).astype(int)
    df = df.loc[:, (df == -1).mean() < NULL_PRECENTAGE]

    # Salva CSV finale per cromosoma
    output_csv = os.path.join(output_folder, f"chr{chr_num}_merged.csv")
    merged_df.to_csv(output_csv)
    print(f"✅ CSV unito salvato in: {output_csv}")
