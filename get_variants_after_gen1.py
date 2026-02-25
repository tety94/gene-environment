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

# -------------------------------
# STEP 1: Concatenazione VCF per generazione
# -------------------------------
concat_files = []

for folder in gen_folders:
    generation_name = os.path.basename(os.path.normpath(folder))

    selected_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith("_selected.vcf.gz")
    ]

    concat_output = os.path.join(output_folder, f"{generation_name}_concat.vcf.gz")
    print(f"🔗 Concatenating {generation_name} cromosomi...")

    subprocess.run(
        ["bcftools", "concat", "-Oz", "-o", concat_output] + selected_files,
        check=True
    )
    subprocess.run(["bcftools", "index", concat_output], check=True)
    concat_files.append((generation_name, concat_output))

# -------------------------------
# STEP 2: Estrazione genotipi in DataFrame
# -------------------------------
all_dfs = []

for gen_name, vcf_file in concat_files:
    print(f"📄 Processing {gen_name} VCF: {vcf_file}")

    cmd = [
        "bcftools", "query",
        "-f", "%CHROM_%POS_%REF_%ALT[\t%GT]\n",
        vcf_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    if not result.stdout.strip():
        print(f"⚠️ Nessuna variante trovata in {vcf_file}, skipping")
        continue

    df = pd.read_csv(StringIO(result.stdout), sep="\t", header=None)
    snp_ids = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    df.columns = snp_ids
    df.index = [f"{gen_name}_sample_{i}" for i in range(df.shape[0])]
    all_dfs.append(df)

# -------------------------------
# STEP 3: Merge dati con outer join
# -------------------------------
print("🔗 Merging all generations with outer join...")
merged_df = pd.concat(all_dfs, axis=0, join="outer")

# -------------------------------
# STEP 4: Gestione missing e binarizzazione
# -------------------------------
# valori mancanti = 0 (variant missing)
merged_df = merged_df.fillna(0)

# binarizzazione: 0 = assenza allele alternativo, 1 = presenza almeno 1 allele alternativo
arr = merged_df.values
arr[arr > 0] = 1
merged_df[:] = arr.astype(np.int8)

# filtra SNP con troppi missing
mask = (merged_df == 0).mean() < NULL_PRECENTAGE
merged_df = merged_df.loc[:, mask]

# rimuove duplicati (campioni)
merged_df = merged_df[~merged_df.index.duplicated(keep="first")]

# -------------------------------
# STEP 5: Salvataggio CSV finale
# -------------------------------
output_csv = os.path.join(output_folder, "significant_variants_merged.csv")
merged_df.to_csv(output_csv)
print(f"✅ CSV finale salvato in: {output_csv}")