import os
import pandas as pd
from cyvcf2 import VCF
from config import VFC_FOLDERS

# -------------------------------
# Configurazione
# -------------------------------
input_folders = VFC_FOLDERS

def genotype_to_numeric(gt):
    """
    Converte un genotipo VCF in numero:
    0/0 -> 0, 0/1 -> 1, 1/1 -> 2, missing -> -1
    """
    if gt is None or gt[0] is None or gt[1] is None:
        return -1
    a, b = gt[0], gt[1]
    if a < 0 or b < 0:
        return -1
    return a + b

def process_vcf_file(vcf_path, output_folder):
    """
    Processa un singolo VCF e salva un CSV per cromosoma
    """
    vcf_file = os.path.basename(vcf_path)
    print(f"  Processing VCF: {vcf_file}")

    vcf = VCF(vcf_path)
    samples = vcf.samples

    # Prepara dizionario per scrivere direttamente riga per riga
    rows = {s: [] for s in samples}
    variants = []

    variant_count = 0
    for variant in vcf:
        alt_allele = variant.ALT[0] if variant.ALT else "."
        var_id = f"{variant.CHROM}_{variant.POS}_{variant.REF}_{alt_allele}"
        variants.append(var_id)

        for i, gt in enumerate(variant.genotypes):
            rows[samples[i]].append(genotype_to_numeric(gt))

        variant_count += 1

    print(f"    {variant_count} varianti lette da {vcf_file}")

    # Crea DataFrame e salva CSV
    df = pd.DataFrame(rows, index=variants).T  # trasponi: righe = campioni, colonne = varianti
    csv_name = os.path.splitext(vcf_file)[0] + "_genotypes.csv"
    output_csv = os.path.join(output_folder, csv_name)
    df.to_csv(output_csv)
    print(f"    ✅ CSV salvato in: {output_csv}\n")

def process_vcf_folder(vcf_folder):
    print(f"Processing folder: {vcf_folder}")
    output_folder = os.path.join(vcf_folder, "genotypes_matrix")
    os.makedirs(output_folder, exist_ok=True)

    vcf_files = [f for f in os.listdir(vcf_folder) if f.endswith("_filtered.vcf")]
    if not vcf_files:
        print(f"⚠️ Nessun VCF trovato in {vcf_folder}")
        return

    for vcf_file in vcf_files:
        vcf_path = os.path.join(vcf_folder, vcf_file)
        process_vcf_file(vcf_path, output_folder)

# -------------------------------
# Ciclo su tutte le cartelle
# -------------------------------
for folder in input_folders:
    process_vcf_folder(os.path.join(folder, 'vcf_filtered'))
