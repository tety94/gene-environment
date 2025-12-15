import os
import pandas as pd
from datetime import datetime
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from config import OUTPUT_FOLDER

merged_folder = OUTPUT_FOLDER
output_csv = os.path.join(merged_folder, "full_chr.csv")

# Elimina il file full_chr.csv se già esiste
if os.path.exists(output_csv):
    os.remove(output_csv)
    print(f"🗑️ File esistente rimosso: {output_csv}")

# Trova tutti i CSV nella cartella (escludendo eventuali file nascosti)
csv_files = [f for f in glob(os.path.join(merged_folder, "*.csv")) if os.path.basename(f) != "full_chr.csv"]

if not csv_files:
    print("⚠️ Nessun CSV trovato nella cartella!")
    exit(1)

print(f"🔹 Trovati {len(csv_files)} CSV da unire.")

# Funzione per leggere un CSV
def read_csv(csv_file):
    print(f"  Leggo {csv_file}")
    df = pd.read_csv(csv_file, index_col=0)
    return df

# Leggi i CSV in parallelo
with ProcessPoolExecutor() as executor:
    dfs = list(executor.map(read_csv, csv_files))

# Merge dei DataFrame in blocchi paralleli
def merge_pairwise(dfs_list):
    """Merge dei dataframe in modo ricorsivo a coppie"""
    while len(dfs_list) > 1:
        new_list = []
        for i in range(0, len(dfs_list), 2):
            if i + 1 < len(dfs_list):
                merged = dfs_list[i].join(dfs_list[i+1], how="outer", rsuffix="_dup")
                # rimuovi colonne duplicate
                merged = merged.loc[:, ~merged.columns.duplicated()]
                new_list.append(merged)
            else:
                new_list.append(dfs_list[i])
        dfs_list = new_list
    return dfs_list[0]

full_df = merge_pairwise(dfs)

# Riempi NaN con -1
full_df = full_df.fillna(-1).astype(int)

# Salva il CSV finale
start_time = datetime.now()
print(f"Inizio a salvare il file alle: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
full_df.to_csv(output_csv)
print(f"✅ CSV completo salvato in: {output_csv}")
end_time = datetime.now()
print(f"Finisco a salvare il file alle: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
