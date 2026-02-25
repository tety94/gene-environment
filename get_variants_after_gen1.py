import os
import subprocess
import pandas as pd

# -------------------------------
# CONFIGURAZIONE
# -------------------------------
gen_folders = [
    "/mnt/cresla_prod/genome_datasets/gen2/",
    "/mnt/cresla_prod/genome_datasets/gen3/"
]

variants_file = "/srv/python-projects/gene-environment/variants_to_extract.csv"
output_folder = "/mnt/cresla_prod/genome_datasets/merged_csv/"
os.makedirs(output_folder, exist_ok=True)

# Legge il file delle varianti
variants_df = pd.read_csv(variants_file, sep=";")
variants_df['variant_id'] = variants_df['chromosome'].astype(str) + "_" + \
                            variants_df['position'].astype(str) + "_" + \
                            variants_df['mutation']

# -------------------------------
# FUNZIONE: estrai varianti per una generazione
# -------------------------------
def extract_variants_for_generation(folder, generation_name):
    print(f"\n🔹 Processing generation: {generation_name}")

    vcf_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith("_selected.vcf.gz")
    ]

    # Lista per accumulare i DataFrame di ogni variante trovata
    variant_dfs = []

    for _, row in variants_df.iterrows():
        chrom = str(row['chromosome'])
        pos = str(row['position'])
        mut = row['mutation']
        var_id = row['variant_id']

        # Trova il VCF relativo al cromosoma
        vcf_path = None
        for vf in vcf_files:
            if f"chr{chrom}" in vf:
                vcf_path = vf
                break
        if not vcf_path:
            continue  # se non c'è VCF per questo cromosoma, salta

        # Costruisce il comando bcftools query
        cmd_query = [
            "bcftools", "query",
            "-f", f"{var_id}[\t%GT]\n",
            "-r", f"{chrom}:{pos}-{pos}",
            vcf_path
        ]

        try:
            result_query = subprocess.run(cmd_query, capture_output=True, text=True, check=True)
            output = result_query.stdout.strip()
            if not output:
                continue  # variante non presente, salta

            # Trasforma in DataFrame
            df_var = pd.DataFrame([x.split("\t")[1:] for x in output.splitlines()])
            df_var.columns = [var_id] * df_var.shape[1]  # Tutte colonne con il nome della variante
            variant_dfs.append(df_var)

        except subprocess.CalledProcessError:
            continue  # ignora errori per varianti mancanti

    # Unisce tutte le varianti trovate
    if variant_dfs:
        df_gen = pd.concat(variant_dfs, axis=1)
        df_gen.index = [f"sample_{i}" for i in range(df_gen.shape[0])]
    else:
        df_gen = pd.DataFrame()

    # Salva CSV della generazione
    output_csv = os.path.join(output_folder, f"{generation_name}_variants.csv")
    df_gen.to_csv(output_csv)
    print(f"✅ CSV generazione salvato in: {output_csv}")
    return df_gen

# -------------------------------
# LOOP su tutte le generazioni
# -------------------------------
gen_csvs = []
for folder in gen_folders:
    gen_name = os.path.basename(os.path.normpath(folder))
    df_gen = extract_variants_for_generation(folder, gen_name)
    gen_csvs.append(df_gen)

# -------------------------------
# UNIONE CSV di gen2 e gen3 con pandas
# -------------------------------
df_merged = pd.concat(gen_csvs, axis=0, join='outer').fillna(0).astype(int)
df_merged.index = [f"sample_{i}" for i in range(df_merged.shape[0])]

merged_csv = os.path.join(output_folder, "significant_variants_merged.csv")
df_merged.to_csv(merged_csv)
print(f"✅ CSV finale unito salvato in: {merged_csv}")