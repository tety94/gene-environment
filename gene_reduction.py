import os
import subprocess
from config import VFC_FOLDERS

# -------------------------------
# Configurazione
# -------------------------------
input_vcf_folders = VFC_FOLDERS

maf_threshold = 0.05       # MAF minima
ld_window_size = 50        # numero SNP per finestra LD
ld_step = 5                # passo della finestra
ld_r2_threshold = 0.2      # soglia r^2 per LD pruning
output_subfolder = "vcf_filtered"  # sottocartella di output

# -------------------------------
# Funzione principale
# -------------------------------
for input_folder in input_vcf_folders:
    output_vcf_folder = os.path.join(input_folder, output_subfolder)
    os.makedirs(output_vcf_folder, exist_ok=True)

    vcf_files = [f for f in os.listdir(input_folder) if f.endswith(".vcf.gz")]

    for vcf_file in vcf_files:
        input_path = os.path.join(input_folder, vcf_file)
        base_name = os.path.splitext(vcf_file)[0]

        # 1️⃣ Converti VCF in file PLINK binario
        plink_prefix = os.path.join(output_vcf_folder, base_name + "_plink")
        subprocess.run([
            "plink2",
            "--vcf", input_path,
            "--make-bed",
            "--out", plink_prefix
        ], check=True)

        # 2️⃣ Filtraggio MAF
        plink_maf_prefix = os.path.join(output_vcf_folder, base_name + "_maf")
        subprocess.run([
            "plink2",
            "--bfile", plink_prefix,
            "--maf", str(maf_threshold),
            "--make-bed",
            "--out", plink_maf_prefix
        ], check=True)

        # 3️⃣ LD pruning
        plink_prune_prefix = os.path.join(output_vcf_folder, base_name + "_pruned")
        subprocess.run([
            "plink2",
            "--bfile", plink_maf_prefix,
            "--indep-pairwise", str(ld_window_size), str(ld_step), str(ld_r2_threshold),
            "--out", plink_prune_prefix
        ], check=True)

        # 4️⃣ Estrazione SNP pruned e salvataggio in VCF finale
        final_vcf_path = os.path.join(output_vcf_folder, base_name + "_filtered.vcf")
        subprocess.run([
            "plink2",
            "--bfile", plink_maf_prefix,
            "--extract", plink_prune_prefix + ".prune.in",
            "--recode", "vcf",
            "--out", os.path.splitext(final_vcf_path)[0]
        ], check=True)

        print(f"✅ File filtrato salvato in: {final_vcf_path}")
