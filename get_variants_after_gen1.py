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

# CSV con le varianti da estrarre
variants_csv = "/srv/python-projects/gene-environment/variant_results_significant.csv"
variants_df = pd.read_csv(variants_csv, sep=";")

# Costruisci lista di varianti in formato CHR_POS_REF_ALT
variants_of_interest = []
for _, row in variants_df.iterrows():
    chrom = str(row["chromosome"])
    pos = str(row["position"])
    ref, alt = row["mutation"].split("_")
    variant_id = f"{chrom}:{pos}-{ref}-{alt}"
    variants_of_interest.append(variant_id)

# -------------------------------
# STEP 1: Estrai varianti per generazione
# -------------------------------
for folder in gen_folders:
    generation_name = os.path.basename(os.path.normpath(folder))
    print(f"🔹 Processing generation: {generation_name}")

    # Lista tutti i VCF selezionati per cromosoma
    vcf_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("_selected.vcf.gz")]
    if not vcf_files:
        print(f"⚠️ Nessun VCF trovato in {folder}, salto generazione")
        continue

    # File VCF temporaneo concatenato
    concat_vcf = os.path.join(output_folder, f"{generation_name}_concat.vcf.gz")
    subprocess.run(["bcftools", "concat", "-Oz", "-o", concat_vcf] + vcf_files, check=True)
    subprocess.run(["bcftools", "index", concat_vcf], check=True)

    # File temporaneo con le varianti da estrarre
    variants_file = os.path.join(output_folder, f"{generation_name}_variants.txt")
    with open(variants_file, "w") as f:
        for v in variants_of_interest:
            f.write(v + "\n")

    # Estrazione varianti in un CSV con Pandas
    cmd = ["bcftools", "query", "-f", "%CHROM_%POS_%REF_%ALT[\t%GT]\n", "-R", variants_file, concat_vcf]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    if not result.stdout.strip():
        print(f"⚠️ Nessuna variante trovata in {generation_name}")
        continue

    df = pd.read_csv(StringIO(result.stdout), sep="\t", header=None)
    snp_ids = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    df.columns = snp_ids
    df.index = [f"{generation_name}_sample_{i}" for i in range(df.shape[0])]

    # Fill missing e binarizzazione
    df = df.fillna(0).astype(int)
    df[df > 0] = 1

    # Salva CSV generazione-specifico
    gen_csv = os.path.join(output_folder, f"{generation_name}_variants.csv")
    df.to_csv(gen_csv)
    print(f"✅ CSV salvato per {generation_name}: {gen_csv}")