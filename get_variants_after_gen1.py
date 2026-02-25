import os
import subprocess
import pandas as pd
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

variants_csv = "/srv/python-projects/gene-environment/variant_results_significant.csv"
variants_df = pd.read_csv(variants_csv, sep=";")

# Costruisci lista di varianti CHR_POS_REF_ALT
variants_of_interest = []
for _, row in variants_df.iterrows():
    chrom = str(row["chromosome"])
    pos = str(row["position"])
    ref, alt = row["mutation"].split("_")
    variants_of_interest.append(f"{chrom}_{pos}_{ref}_{alt}")

# -------------------------------
# STEP 1: Estrai varianti per generazione
# -------------------------------
gen_csv_paths = []

for folder in gen_folders:
    generation_name = os.path.basename(os.path.normpath(folder))
    print(f"🔹 Processing generation: {generation_name}")

    # Lista VCF per cromosoma
    vcf_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("_selected.vcf.gz")]
    if not vcf_files:
        print(f"⚠️ Nessun VCF trovato in {folder}, salto generazione")
        continue

    # Concatena tutti i VCF cromosoma
    concat_vcf = os.path.join(output_folder, f"{generation_name}_concat.vcf.gz")
    subprocess.run(["bcftools", "concat", "-Oz", "-o", concat_vcf] + vcf_files, check=True)
    subprocess.run(["bcftools", "index", concat_vcf], check=True)

    # -------------------------------
    # Ciclo variante per variante
    # -------------------------------
    dfs = []
    for var in variants_of_interest:
        chrom, pos, ref, alt = var.split("_")
        try:
            cmd = [
                "bcftools",
                "query",
                "-f", f"{var}[\t%GT]\n",
                "-r", f"{chrom}:{pos}-{pos}",
                concat_vcf
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                continue  # variante non trovata
            df_var = pd.read_csv(StringIO(result.stdout), sep="\t", header=None)
            df_var.columns = [var]
            dfs.append(df_var)
        except subprocess.CalledProcessError:
            # Se bcftools fallisce (variante non trovata), ignoriamo
            continue

    if not dfs:
        print(f"⚠️ Nessuna variante trovata per {generation_name}")
        continue

    # Concatenazione delle colonne (varianti) → campioni come righe
    df_gen = pd.concat(dfs, axis=1)
    df_gen.index = [f"{generation_name}_sample_{i}" for i in range(df_gen.shape[0])]

    # Binarizzazione: 0 = assenza allele alternativo, 1 = presenza almeno 1 allele
    df_gen = df_gen.fillna(0).astype(int)
    df_gen[df_gen > 0] = 1

    # Salva CSV generazione-specifico
    gen_csv = os.path.join(output_folder, f"{generation_name}_variants.csv")
    df_gen.to_csv(gen_csv)
    gen_csv_paths.append(gen_csv)
    print(f"✅ CSV salvato per {generation_name}: {gen_csv}")