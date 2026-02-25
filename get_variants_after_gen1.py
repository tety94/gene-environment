import pandas as pd
import os
import subprocess

# -------------------------
# CONFIG
# -------------------------

csv_file = "variants_from_db.csv"
gen_folders = [
    "/mnt/cresla_prod/genome_datasets/gen2/",
    "/mnt/cresla_prod/genome_datasets/gen3/"
]

regions_file = "regions.txt"
final_output = "significant_variants_merged.vcf.gz"

# -------------------------
# Step 1: Leggi varianti dal CSV
# -------------------------

df = pd.read_csv(csv_file, sep=";")

# mutation è tipo A_G
df[['REF', 'ALT']] = df['mutation'].str.split('_', expand=True)

# crea file region per bcftools (CHR POS POS)
with open(regions_file, "w") as f:
    for _, row in df.iterrows():
        f.write(f"{row['chromosome']}\t{row['position']}\t{row['position']}\n")

print("✅ regions.txt creato")

# -------------------------
# Step 2: Estrazione per ogni VCF
# -------------------------

temp_files = []

for folder in gen_folders:
    for file in os.listdir(folder):
        if file.endswith(".vcf.gz") and not file.endswith(".tbi"):
            input_vcf = os.path.join(folder, file)
            output_vcf = input_vcf.replace(".vcf.gz", "_selected.vcf.gz")

            print(f"🔎 Estrazione da {input_vcf}")

            subprocess.run([
                "bcftools",
                "view",
                "-R", regions_file,
                input_vcf,
                "-Oz",
                "-o", output_vcf
            ], check=True)

            subprocess.run(["bcftools", "index", output_vcf], check=True)

            temp_files.append(output_vcf)

# -------------------------
# Step 3: Merge campioni
# -------------------------

print("🔗 Merge finale...")

subprocess.run([
                   "bcftools",
                   "merge",
                   "-Oz",
                   "-o", final_output
               ] + temp_files, check=True)

subprocess.run(["bcftools", "index", final_output], check=True)

print(f"✅ File finale creato: {final_output}")