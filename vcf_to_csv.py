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


def process_vcf_folder(vcf_folder):
    print(f"Processing folder: {vcf_folder}")

    output_folder = os.path.join(vcf_folder, "genotypes_matrix")
    os.makedirs(output_folder, exist_ok=True)
    output_csv = os.path.join(output_folder, "genotypes_matrix.csv")

    # Trova tutti i VCF filtrati
    vcf_files = [f for f in os.listdir(vcf_folder) if f.endswith("_filtered.vcf")]
    if not vcf_files:
        print(f"⚠️ Nessun VCF trovato in {vcf_folder}")
        return

    genotypes_dict = {}
    all_variants = []

    for vcf_file in vcf_files:
        vcf_path = os.path.join(vcf_folder, vcf_file)
        vcf = VCF(vcf_path)
        samples = vcf.samples

        for s in samples:
            if s not in genotypes_dict:
                genotypes_dict[s] = {}

        variant_count = 0
        for variant in vcf:
            # Prendi il primo ALT se multiallele
            alt_allele = variant.ALT[0] if variant.ALT else "."
            var_id = f"{variant.CHROM}_{variant.POS}_{variant.REF}_{alt_allele}"
            all_variants.append(var_id)

            for i, gt in enumerate(variant.genotypes):
                genotypes_dict[samples[i]][var_id] = genotype_to_numeric(gt)
            variant_count += 1

        print(f"  {variant_count} varianti lette da {vcf_file}")

    # Creazione DataFrame
    all_variants = sorted(list(set(all_variants)))
    df = pd.DataFrame(index=genotypes_dict.keys(), columns=all_variants, dtype=int)

    for sample, geno_dict in genotypes_dict.items():
        for var in all_variants:
            df.at[sample, var] = geno_dict.get(var, -1)

    df.to_csv(output_csv)
    print(f"✅ Matrice genotipi salvata in: {output_csv}\n")


# -------------------------------
# Ciclo su tutte le cartelle
# -------------------------------
for folder in input_folders:
    process_vcf_folder(folder + '/vcf_filtered')
