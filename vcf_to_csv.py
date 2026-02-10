import os
import pandas as pd
from cyvcf2 import VCF
from concurrent.futures import ProcessPoolExecutor
from config import VFC_FOLDERS, MAX_WORKERS

# -------------------------------
# Configurazione
# -------------------------------
input_folders = VFC_FOLDERS  # Lista di cartelle contenenti file VCF già filtrati (_filtered.vcf)

# -------------------------------
# Funzioni di supporto
# -------------------------------

def genotype_to_numeric(gt):
    """
    Converte un genotipo in formato numerico:
    - gt è una lista di alleli [allele1, allele2, phased]
    - Restituisce la somma degli alleli (0=omozigote ref, 1=eterozigote, 2=omozigote alt)
    - Se il genotipo è mancante, restituisce -1
    """
    if gt is None or gt[0] is None or gt[1] is None:
        return -1
    a, b = gt[0], gt[1]
    if a < 0 or b < 0:
        return -1
    return a + b

def process_vcf_file(vcf_path, output_folder):
    """
    Processa un singolo VCF e salva un CSV con la matrice genotipi:
    - Righe: campioni
    - Colonne: varianti (SNP)
    - Valori: 0,1,2 per numero di alleli alternativi
    """
    vcf_file = os.path.basename(vcf_path)
    print(f"[PID {os.getpid()}] Processing VCF: {vcf_file}")

    # Legge il VCF usando cyvcf2
    vcf = VCF(vcf_path)
    samples = vcf.samples  # Lista dei campioni nel VCF

    # Dizionario per accumulare genotipi per ciascun campione
    rows = {s: [] for s in samples}
    variants = []  # Lista dei nomi degli SNP (chrom_pos_ref_alt)

    variant_count = 0
    for variant in vcf:
        # Nome univoco della variante
        alt_allele = variant.ALT[0] if variant.ALT else "."
        var_id = f"{variant.CHROM}_{variant.POS}_{variant.REF}_{alt_allele}"
        variants.append(var_id)

        # Converte i genotipi di tutti i campioni in numerico
        for i, gt in enumerate(variant.genotypes):
            rows[samples[i]].append(genotype_to_numeric(gt))

        variant_count += 1

    print(f"  {variant_count} varianti lette da {vcf_file}")

    # Crea il dataframe pandas: righe = campioni, colonne = SNP
    df = pd.DataFrame(rows, index=variants).T

    # Salva il CSV
    csv_name = os.path.splitext(vcf_file)[0] + "_genotypes.csv"
    output_csv = os.path.join(output_folder, csv_name)
    df.to_csv(output_csv)
    print(f"  ✅ CSV salvato in: {output_csv}")

def process_vcf_folder(vcf_folder):
    """
    Processa tutti i VCF di una cartella:
    - Crea una sottocartella "genotypes_matrix" per salvare i CSV
    - Parallelizza l'elaborazione dei VCF usando più processi
    """
    print(f"Processing folder: {vcf_folder}")
    output_folder = os.path.join(vcf_folder, "genotypes_matrix")
    os.makedirs(output_folder, exist_ok=True)

    # Trova tutti i VCF filtrati (_filtered.vcf) nella cartella
    vcf_files = [os.path.join(vcf_folder, f) for f in os.listdir(vcf_folder) if f.endswith("_filtered.vcf")]
    if not vcf_files:
        print(f"⚠️ Nessun VCF trovato in {vcf_folder}")
        return

    # Parallelizza i VCF usando ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_vcf_file, vcf_path, output_folder) for vcf_path in vcf_files]
        for f in futures:
            f.result()  # attende che ogni VCF finisca

# -------------------------------
# Ciclo su tutte le cartelle di input
# -------------------------------
for folder in input_folders:
    # Processa la sottocartella "vcf_filtered" di ogni cartella principale
    process_vcf_folder(os.path.join(folder, 'vcf_filtered'))
