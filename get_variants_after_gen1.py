import os
import pandas as pd
import numpy as np
import subprocess
from glob import glob

# -------------------------------
# CONFIGURAZIONE
# -------------------------------
gen_folders = [
    "/mnt/cresla_prod/genome_datasets/gen2/",
    "/mnt/cresla_prod/genome_datasets/gen3/"
]

variants_csv = "variants_from_db.csv"  # CSV con varianti significative dal DB
output_folder = "/mnt/cresla_prod/genome_datasets/merged_csv/"
NULL_PRECENTAGE = 0.1
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# STEP 1: Leggi varianti da estrarre
# -------------------------------
df_var = pd.read_csv(variants_csv, sep=";")
df_var[['REF', 'ALT']] = df_var['mutation'].str.split('_', expand=True)

# Creiamo file regions.txt per bcftools (CHR POS POS)
regions_file = os.path.join(output_folder, "regions.txt")
with open(regions_file, "w") as f:
    for _, row in df_var.iterrows():
        f.write(f"{row['chromosome']}\t{row['position']}\t{row['position']}\n")

# -------------------------------
# STEP 2: Estrazione varianti dai VCF
# -------------------------------
all_dfs = []

for folder in gen_folders:
    generation_name = os.path.basename(os.path.normpath(folder))
    print(f"\n🔹 Processing generation: {generation_name}")

    vcf_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".vcf.gz")]

    for vcf_file in vcf_files:
        chr_name = os.path.basename(vcf_file).split(".")[0]
        print(f"  Processing {chr_name}")

        # File temporaneo per VCF filtrato
        tmp_vcf = os.path.join(output_folder, f"{chr_name}_selected.vcf.gz")

        # bcftools view per estrarre solo le varianti di interesse
        subprocess.run([
            "bcftools", "view", "-R", regions_file,
            "-Oz", "-o", tmp_vcf, vcf_file
        ], check=True)
        subprocess.run(["bcftools", "index", tmp_vcf], check=True)

        # Leggi il VCF estratto in DataFrame
        cmd = [
            "bcftools", "query",
            "-f", "%CHROM_%POS_%REF_%ALT[\t%GT]\n",
            tmp_vcf
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        df_chr = pd.read_csv(
            pd.compat.StringIO(result.stdout),
            sep="\t",
            header=None
        )

        # Prima colonna = SNP ID
        snp_ids = df_chr.iloc[:, 0]
        df_chr = df_chr.iloc[:, 1:]
        df_chr.columns = snp_ids

        # Righe = campioni (puoi sostituire con sample ID reali se vuoi)
        df_chr.index = [f"sample_{i}" for i in range(df_chr.shape[0])]

        all_dfs.append(df_chr)

# -------------------------------
# STEP 3: Merge cromosomi e generazioni
# -------------------------------
merged_df = pd.concat(all_dfs, axis=1, join="outer")

# -------------------------------
# STEP 4: Gestione missing e binarizzazione
# -------------------------------
merged_df = merged_df.fillna(-1).astype(int)

# Filtra SNP con troppi missing
merged_df = merged_df.loc[:, (merged_df == -1).mean() < NULL_PRECENTAGE]

# Binaria genotipi: 0 = no allele alt, 1 = almeno 1 allele alt
arr = merged_df.values
arr[arr < 0] = 0
arr[arr > 0] = 1
merged_df[:] = arr.astype(np.int8)

# -------------------------------
# STEP 5: Rimozione duplicati (campioni ripetuti)
# -------------------------------
merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

# -------------------------------
# STEP 6: Salvataggio CSV finale
# -------------------------------
output_csv = os.path.join(output_folder, "significant_variants_merged.csv")
merged_df.to_csv(output_csv)
print(f"\n✅ CSV finale salvato in: {output_csv}")