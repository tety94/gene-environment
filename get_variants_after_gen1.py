import os
import subprocess

gen_folders = [
    "/mnt/cresla_prod/genome_datasets/gen2/",
    "/mnt/cresla_prod/genome_datasets/gen3/"
]

final_files = []

for folder in gen_folders:
    selected_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith("_selected.vcf.gz")
    ]

    generation_name = os.path.basename(os.path.normpath(folder))

    concat_output = f"{generation_name}_concat.vcf.gz"
    filtered_output = f"{generation_name}_noACH.vcf.gz"

    print(f"🔗 Concat {generation_name}")

    subprocess.run([
                       "bcftools",
                       "concat",
                       "-Oz",
                       "-o", concat_output
                   ] + selected_files, check=True)

    subprocess.run(["bcftools", "index", concat_output], check=True)

    print(f"🚫 Rimozione ACH da {generation_name}")

    subprocess.run([
        "bcftools",
        "view",
        "-s", "^ACH*",
        concat_output,
        "-Oz",
        "-o", filtered_output
    ], check=True)

    subprocess.run(["bcftools", "index", filtered_output], check=True)

    final_files.append(filtered_output)

# Merge finale tra generazioni
print("🔗 Merge finale gen2 + gen3")

subprocess.run([
                   "bcftools",
                   "merge",
                   "-Oz",
                   "-o", "significant_variants_final.vcf.gz"
               ] + final_files, check=True)

subprocess.run(["bcftools", "index", "significant_variants_final.vcf.gz"], check=True)

print("✅ File finale creato: significant_variants_final.vcf.gz")