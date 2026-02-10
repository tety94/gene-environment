import os
import subprocess
from config import VFC_FOLDERS

# -------------------------------
# Configurazione
# -------------------------------
input_vcf_folders = VFC_FOLDERS  # Cartelle contenenti i file VCF da processare

maf_threshold = 0.001             # Frequenza minima dell'allele minore (MAF) per filtrare SNP troppo rari
ld_window_size = 50               # Numero di SNP considerati in ogni finestra per LD pruning
ld_step = 5                        # Passo della finestra per il calcolo del LD
ld_r2_threshold = 0.8              # Soglia r^2 per rimuovere SNP ridondanti in LD
output_subfolder = "vcf_filtered"  # Sottocartella in cui salvare i file filtrati

# -------------------------------
# Ciclo principale sui folder
# -------------------------------
for input_folder in input_vcf_folders:
    # Creazione della cartella di output se non esiste
    output_vcf_folder = os.path.join(input_folder, output_subfolder)
    os.makedirs(output_vcf_folder, exist_ok=True)

    # Lista dei file VCF compressi (.vcf.gz) presenti nella cartella
    vcf_files = [f for f in os.listdir(input_folder) if f.endswith(".vcf.gz") and not f.startswith("._")]

    for vcf_file in vcf_files:
        input_path = os.path.join(input_folder, vcf_file)
        base_name = os.path.splitext(vcf_file)[0]

        # 1️⃣ Converti il VCF in formato PLINK binario (.bed/.bim/.fam)
        # Questo formato è più efficiente per le analisi di genetica statistica
        plink_prefix = os.path.join(output_vcf_folder, base_name + "_plink")
        subprocess.run([
            "plink2",
            "--vcf", input_path,
            "--make-bed",
            "--out", plink_prefix
        ], check=True)

        # 2️⃣ Filtraggio SNP per frequenza dell'allele minore (MAF)
        # Rimuove SNP troppo rari (MAF < 0.001), che possono introdurre rumore statistico
        plink_maf_prefix = os.path.join(output_vcf_folder, base_name + "_maf")
        subprocess.run([
            "plink2",
            "--bfile", plink_prefix,
            "--maf", str(maf_threshold),
            "--make-bed",
            "--out", plink_maf_prefix
        ], check=True)

        # 3️⃣ LD pruning (Linkage Disequilibrium)
        # Rimuove SNP altamente correlati tra loro (r^2 > 0.8) mantenendo solo SNP relativamente indipendenti
        # Parametri: finestra di 50 SNP, passo 5 SNP, soglia r^2 0.8
        plink_prune_prefix = os.path.join(output_vcf_folder, base_name + "_pruned")
        subprocess.run([
            "plink2",
            "--bfile", plink_maf_prefix,
            "--indep-pairwise", str(ld_window_size), str(ld_step), str(ld_r2_threshold),
            "--out", plink_prune_prefix
        ], check=True)

        # 4️⃣ Estrazione dei SNP filtrati e salvataggio in VCF finale
        # Manteniamo solo gli SNP selezionati dal LD pruning
        final_vcf_path = os.path.join(output_vcf_folder, base_name + "_filtered.vcf")
        subprocess.run([
            "plink2",
            "--bfile", plink_maf_prefix,
            "--extract", plink_prune_prefix + ".prune.in",
            "--recode", "vcf",
            "--out", os.path.splitext(final_vcf_path)[0]
        ], check=True)

        # Messaggio di conferma
        print(f"✅ File filtrato salvato in: {final_vcf_path}")
