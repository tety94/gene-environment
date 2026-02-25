import os
import subprocess
import pandas as pd
import numpy as np
from io import StringIO

# -------------------------------
# CONFIGURAZIONE
# -------------------------------
gen_folders = [
    "/mnt/cresla_prod/genome_datasets/gen2/",
    "/mnt/cresla_prod/genome_datasets/gen3/"
]

output_folder = "/mnt/cresla_prod/genome_datasets/merged_csv/"
os.makedirs(output_folder, exist_ok=True)
NULL_PRECENTAGE = 0.1

# CSV con le varianti da prendere (una variante per riga, formato CHR_POS_REF_ALT)
variants_file = "/srv/python-projects/gene-environment/variants_to_extract.csv"
variants = pd.read_csv(variants_file, header=None)[0].tolist()

# -------------------------------
# FUNZIONE PER ESTRARRE VARIANTI
# -------------------------------
def extract_variants_for_generation(gen_folder, generation_name):
    print(f"\n🔹 Processing generation: {generation_name}")
    concat_vcfs = []

    # Trova tutti i VCF selezionati per cromosoma
    for f in os.listdir(gen_folder):
        if f.endswith("_selected.vcf.gz"):
            concat_vcfs.append(os.path.join(gen_folder, f))

    dfs = []

    for var in variants:
        chrom, pos, ref, alt = var.split("_")
        pos = int(pos)

        for vcf in concat_vcfs:
            try:
                cmd = [
                    "bcftools",
                    "query",
                    "-f", f"{var}[\t%GT]\n",
                    "-r", f"{chrom}:{pos}-{pos}",
                    vcf
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if not result.stdout.strip():
                    continue  # variante non trovata in questo VCF
                df_var = pd.read_csv(StringIO(result.stdout), sep="\t", header=None)
                df_var = df_var.T               # righe = campioni, colonna = SNP
                df_var.columns = [var]
                dfs.append(df_var)
            except subprocess.CalledProcessError:
                continue

    if not dfs:
        print(f"⚠️ Nessuna variante trovata per {generation_name}")
        return None

    merged_df = pd.concat(dfs, axis=1, sort=False)
    merged_df = merged_df.fillna(-1).astype(int)

    # Filtra SNP con troppi missing
    merged_df = merged_df.loc[:, (merged_df == -1).mean() < NULL_PRECENTAGE]

    # Binarizza genotipi: 0 = assenza allele alternativo, 1 = presenza almeno 1 allele alternativo
    arr = merged_df.values
    arr[arr < 0] = 0
    arr[arr > 0] = 1
    merged_df[:] = arr.astype(np.int8)

    # Rimuove duplicati (campioni)
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

    output_csv = os.path.join(output_folder, f"{generation_name}_variants.csv")
    merged_df.to_csv(output_csv)
    print(f"✅ CSV per {generation_name} salvato in: {output_csv}")
    return output_csv

# -------------------------------
# ESECUZIONE PER LE GENERAZIONI
# -------------------------------
csv_files = []
for folder in gen_folders:
    generation_name = os.path.basename(os.path.normpath(folder))
    csv_file = extract_variants_for_generation(folder, generation_name)
    if csv_file:
        csv_files.append(csv_file)

print("\n📄 Tutti i CSV generati. Ora puoi unirli con pandas se vuoi.")