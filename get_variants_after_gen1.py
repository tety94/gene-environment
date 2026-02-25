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

# Costruisci lista di varianti in formato CHR:POS-REF-ALT
variants_of_interest = []
for _, row in variants_df.iterrows():
    chrom = str(row["chromosome"])
    pos = str(row["position"])
    ref, alt = row["mutation"].split("_")
    variants_of_interest.append(f"{chrom}:{pos}-{ref}-{alt}")

# -------------------------------
# STEP 1: Estrai varianti per generazione
# -------------------------------
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
    # Legge tutte le varianti presenti nel VCF
    # -------------------------------
    cmd_sites = ["bcftools", "query", "-f", "%CHROM:%POS-%REF-%ALT\n", concat_vcf]
    result_sites = subprocess.run(cmd_sites, capture_output=True, text=True, check=True)
    vcf_sites = set(result_sites.stdout.strip().split("\n"))

    # Intersezione con varianti di interesse (ignora quelle non presenti)
    selected_sites = [v for v in variants_of_interest if v in vcf_sites]
    if not selected_sites:
        print(f"⚠️ Nessuna variante di interesse trovata in {generation_name}")
        continue

    # File temporaneo con varianti presenti
    variants_file = os.path.join(output_folder, f"{generation_name}_variants_present.txt")
    with open(variants_file, "w") as f:
        for v in selected_sites:
            f.write(v + "\n")

    # -------------------------------
    # Estrazione genotipi e creazione CSV
    # -------------------------------
    cmd_query = ["bcftools", "query", "-f", "%CHROM_%POS_%REF_%ALT[\t%GT]\n", "-R", variants_file, concat_vcf]
    result_query = subprocess.run(cmd_query, capture_output=True, text=True, check=True)

    df = pd.read_csv(StringIO(result_query.stdout), sep="\t", header=None)
    snp_ids = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    df.columns = snp_ids
    df.index = [f"{generation_name}_sample_{i}" for i in range(df.shape[0])]

    # Binarizza genotipi: 0 = assenza allele alternativo, 1 = presenza almeno 1 allele
    df = df.fillna(0).astype(int)
    df[df > 0] = 1

    # Salva CSV generazione-specifico
    gen_csv = os.path.join(output_folder, f"{generation_name}_variants.csv")
    df.to_csv(gen_csv)
    print(f"✅ CSV salvato per {generation_name}: {gen_csv}")