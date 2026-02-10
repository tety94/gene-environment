import os
from glob import glob
from config import OUTPUT_FOLDER

# Cartella contenente i CSV dei cromosomi già uniti e binarizzati
merged_folder = OUTPUT_FOLDER

# Nome del file finale che conterrà tutti i cromosomi uniti
output_file = os.path.join(merged_folder, "full_chr.csv")

# -------------------------------
# Elimina il file finale se già esiste
# -------------------------------
if os.path.exists(output_file):
    os.remove(output_file)

# Trova tutti i CSV dei cromosomi (chr1_merged.csv, chr2_merged.csv, ...)
# e ordina alfabeticamente
csv_files = sorted([
    f for f in glob(os.path.join(merged_folder, "chr*_merged.csv"))
    if not f.endswith("full_chr.csv")  # esclude eventuali file precedenti full_chr.csv
])

# -------------------------------
# Funzione di controllo: verifica che gli ID dei campioni siano coerenti
# -------------------------------
def get_all_ids(fpath):
    """
    Legge tutti gli ID dei campioni da un CSV (prima colonna),
    saltando l'intestazione.
    """
    with open(fpath) as f:
        next(f)  # salta header
        return [line.split(",")[0] for line in f]

# Prende come riferimento gli ID dal primo CSV
ref_ids = get_all_ids(csv_files[0])

# Controlla che tutti gli altri CSV abbiano gli stessi ID dei campioni
for f in csv_files[1:]:
    ids = get_all_ids(f)
    if ids != ref_ids:
        # Se gli ID non corrispondono, interrompe l'esecuzione
        raise RuntimeError(f"❌ ID mismatch in {f}")

print(f"🔹 Unione testuale di {len(csv_files)} file")

# -------------------------------
# COSTRUZIONE HEADER FINALE
# -------------------------------
# Legge header del primo CSV
with open(csv_files[0], "r") as f:
    header = f.readline().strip()

# Aggiunge i nomi delle colonne dei cromosomi successivi, saltando la colonna ID
for fpath in csv_files[1:]:
    with open(fpath, "r") as f:
        cols = f.readline().strip().split(",")[1:]  # salta colonna ID
        header += "," + ",".join(cols)

# Scrive l'header finale nel file di output
with open(output_file, "w") as out:
    out.write(header + "\n")

# -------------------------------
# UNIONE DELLE RIGHE (VALORI GENOTIPICI)
# -------------------------------
# Apre tutti i file CSV
files = [open(f, "r") for f in csv_files]

# Salta l'intestazione di ciascun CSV
for f in files:
    f.readline()

# Scrive i dati nel file finale, combinando le colonne dei diversi cromosomi
with open(output_file, "a") as out:
    # zip(*files) prende una riga da ogni file contemporaneamente
    for rows in zip(*files):
        # La prima colonna è l'ID del campione
        base = rows[0].strip()
        rest = []
        # Per ogni file successivo, prendi solo i genotipi (salta colonna ID)
        for r in rows[1:]:
            rest.append(",".join(r.strip().split(",")[1:]))
        # Scrive riga completa: ID + genotipi di tutti i cromosomi
        out.write(base + "," + ",".join(rest) + "\n")

# Chiudi tutti i file aperti
for f in files:
    f.close()

print(f"✅ File finale creato: {output_file}")
