import os
import pandas as pd
from cyvcf2 import VCF
from config import VFC_FOLDERS

# -------------------------------
# Configurazione cartelle
# -------------------------------
input_folders = VFC_FOLDERS


# -------------------------------
# Funzione per generare matrice genotipi per una cartella
# -------------------------------
def generate_genotype_matrix(vcf_folder):
    output_folder = os.path.join(vcf_folder, "genotypes_matrix")
    os.makedirs(output_folder, exist_ok=True)
    output_csv = os.path.join(output_folder, "genotypes_matrix.csv")

    # Lista dei VCF finali filtrati
    vcf_files = [f for f in os.listdir(vcf_folder) if f.endswith("_filtered.vcf")]

    genotypes_dict = {}
    all_variants = []

    for vcf_file in vcf_files:
        vcf_path = os.path.join(vcf_folder, vcf_file)
        vcf = VCF(vcf_path)

        samples = vcf.samples
        for s in samples:
            if s not in genotypes_dict:
                genotypes_dict[s] = {}

        for variant in vcf:
            var_id = f"{variant.CHROM}_{variant.POS}_{variant.REF}_{variant.ALT[0]}"
            all_variants.append(var_id)

            for i, gt in enumerate(variant.genotypes):
                if gt[0] is None or gt[1] is None:
                    g = -1
                else:
                    g = gt[0] + gt[1]
                genotypes_dict[samples[i]][var_id] = g

    # Creazione DataFrame
    all_variants = sorted(list(set(all_variants)))
    df = pd.DataFrame(index=genotypes_dict.keys(), columns=all_variants)

    for sample, geno_dict in genotypes_dict.items():
        for var in all_variants:
            df.at[sample, var] = geno_dict.get(var, -1)

    df.to_csv(output_csv)
    print(f"✅ Matrice genotipi salvata in: {output_csv}")


# -------------------------------
# Ciclo su tutte le cartelle
# -------------------------------
for folder in input_folders:
    generate_genotype_matrix(folder)
