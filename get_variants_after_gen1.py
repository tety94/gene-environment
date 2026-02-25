import os
import subprocess
import pandas as pd
import numpy as np

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

# -------------------------------
# STEP 1: Concatenazione VCF per generazione
# -------------------------------
final_files = []

for folder in gen_folders:
    generation_name = os.path.basename(os.path.normpath(folder))

    selected_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith("_selected.vcf.gz")
    ]

    concat_output = os.path.join(output_folder, f"{generation_name}_concat.vcf.gz")

    print(f"🔗 Concat {generation_name} cromosomi")

    subprocess.run([
                       "bcftools",
                       "concat",
                       "-Oz",
                       "-o", concat_output
                   ] + selected_files, check=True)

    subprocess.run(["bcftools", "index", concat_output], check=True)

    final_files.append(concat_output)

# -------------------------------
# STEP 2: Merge finale tra generazioni
# -------------------------------
final_vcf = os.path.join(output_folder, "significant_variants_merged.vcf.gz")
print("🔗 Merge finale gen2 + gen3")

subprocess.run([
                   "bcftools",
                   "merge",
                   "-Oz",
                   "-o", final_vcf
               ] + final_files, check=True)

subprocess.run(["bcftools", "index", final_vcf], check=True)
print(f"✅ VCF finale creato: {final_vcf}")

# -------------------------------
# STEP 3: Trasforma VCF finale in CSV
# -------------------------------
print("📄 Conversione VCF in CSV binarizzato...")

# Legge tutte le varianti
cmd = [
    "bcftools",
    "query",
    "-f", "%CHROM_%POS_%REF_%ALT[\t%GT]\n",
    final_vcf
]

result = subprocess.run(cmd, capture_output=True, text=True, check=True)

# Se non ci sono varianti, esce
if not result.stdout.strip():
    raise ValueError("❌ Nessuna variante trovata nel VCF finale!")

# Trasforma in DataFrame
from io import StringIO

df = pd.read_csv(StringIO(result.stdout), sep="\t", header=None)

# Colonna 0 = SNP ID
snp_ids = df.iloc[:, 0]
df = df.iloc[:, 1:]
df.columns = snp_ids

# Campioni come righe
df.index = [f"sample_{i}" for i in range(df.shape[0])]

# -------------------------------
# STEP 4: Gestione missing e binarizzazione
# -------------------------------
df = df.fillna(-1).astype(int)

# Filtra SNP con troppi missing
df = df.loc[:, (df == -1).mean() < NULL_PRECENTAGE]

# Binarizza genotipi: 0 = assenza allele alternativo, 1 = presenza almeno 1 allele alternativo
arr = df.values
arr[arr < 0] = 0
arr[arr > 0] = 1
df[:] = arr.astype(np.int8)

# Rimuove duplicati (campioni)
df = df[~df.index.duplicated(keep='first')]

# -------------------------------
# STEP 5: Salvataggio CSV finale
# -------------------------------
output_csv = os.path.join(output_folder, "significant_variants_merged.csv")
df.to_csv(output_csv)
print(f"✅ CSV finale salvato in: {output_csv}")