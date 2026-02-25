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

# Varianti da selezionare (esempio: prendi dal DB o lista)
variants_of_interest = [
    "2_220623450_A_G",
    "3_102187338_G_A",
    "8_22566050_G_A",
    "8_29896508_T_C",
    "4_44791971_G_T"
]

# -------------------------------
# STEP 1: Filtra VCF per generazione
# -------------------------------
generation_vcfs = {}

for folder in gen_folders:
    generation_name = os.path.basename(os.path.normpath(folder))
    print(f"🔹 Processing generation: {generation_name}")

    # Lista VCF per cromosoma
    vcf_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("_selected.vcf.gz")]

    # File VCF unico per la generazione
    gen_vcf_path = os.path.join(output_folder, f"{generation_name}_variants.vcf.gz")

    # Costruisco il comando bcftools per estrarre solo le varianti di interesse
    cmd = [
        "bcftools", "view",
        "-Oz",
        "-o", gen_vcf_path
    ]

    # Aggiungo tutti i VCF cromosoma
    cmd += vcf_files

    # Estrazione varianti tramite -T <(echo variants)
    # Creiamo un file temporaneo con le varianti
    variants_file = os.path.join(output_folder, f"{generation_name}_variants.txt")
    with open(variants_file, "w") as f:
        for v in variants_of_interest:
            # Formato CHR:POS-REF-ALT per bcftools
            chrom, pos, ref, alt = v.split("_")
            f.write(f"{chrom}:{pos}-{ref}-{alt}\n")

    # Aggiungo il filtro
    cmd += ["-T", variants_file]

    print(f"  🔗 Extracting selected variants for {generation_name}")
    subprocess.run(cmd, check=True)
    subprocess.run(["bcftools", "index", gen_vcf_path], check=True)

    generation_vcfs[generation_name] = gen_vcf_path

# -------------------------------
# STEP 2: Leggi VCF generazione-specifici e unisci in Pandas
# -------------------------------
dfs = []
for gen, vcf_path in generation_vcfs.items():
    print(f"📄 Reading {gen} VCF into Pandas")

    cmd = ["bcftools", "query", "-f", "%CHROM_%POS_%REF_%ALT[\t%GT]\n", vcf_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    if not result.stdout.strip():
        print(f"⚠️ Nessuna variante trovata in {vcf_path}")
        continue

    df = pd.read_csv(StringIO(result.stdout), sep="\t", header=None)
    snp_ids = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    df.columns = snp_ids
    df.index = [f"{gen}_sample_{i}" for i in range(df.shape[0])]
    dfs.append(df)

# Merge esterno per includere tutte le varianti
merged_df = pd.concat(dfs, axis=0, join="outer").fillna(0).astype(int)

# Binarizzazione: 0 = assenza allele alternativo, 1 = presenza almeno 1 allele alternativo
merged_df[merged_df > 0] = 1

# Salvataggio CSV finale
output_csv = os.path.join(output_folder, "significant_variants_merged.csv")
merged_df.to_csv(output_csv)
print(f"✅ CSV finale salvato in: {output_csv}")