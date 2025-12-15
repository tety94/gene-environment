import os
import pandas as pd
from glob import glob
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

# Leggi tutti i CSV e mettili in una lista
dfs = []
for csv_file in csv_files:
    print(f"  Leggo {csv_file}")
    df = pd.read_csv(csv_file, index_col=0)
    dfs.append(df)

# Merge progressivo
full_df = dfs[0]
for df in dfs[1:]:
    full_df = full_df.join(df, how="outer", rsuffix="_dup")  # outer join per tenere tutti i campioni
    # Rimuovi eventuali colonne duplicate
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]

# Riempi NaN con -1
full_df = full_df.fillna(-1).astype(int)

# Salva il CSV finale
full_df.to_csv(output_csv)
print(f"✅ CSV completo salvato in: {output_csv}")
