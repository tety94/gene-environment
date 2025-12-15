#!/usr/bin/env python3
import pandas as pd
from cyvcf2 import VCF
from intervaltree import Interval, IntervalTree
import glob, os, sys, requests

if len(sys.argv) < 2:
    print("Uso: python estrai_varianti.py GENE1 [GENE2 ...]")
    sys.exit(1)

genes_input = sys.argv[1:]
print("Geni richiesti:", genes_input)

# ---- Recupera coordinate ENSEMBL ----
def get_variant_coordinates(gene):
    url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene}?content-type=application/json"
    r = requests.get(url)
    r.raise_for_status()
    d = r.json()
    return d["seq_region_name"], d["start"], d["end"]

regions_by_chr = {}
for variant in genes_input:
    chr_num, s, e = get_variant_coordinates(variant)
    regions_by_chr.setdefault(chr_num, IntervalTree()).add(Interval(s, e+1, variant))
    print(f"{variant}: chr{chr_num}:{s}-{e}")

vcf_folders = [
    "/mnt/cresla_prod/genome_datasets/gen2",
    "/mnt/cresla_prod/genome_datasets/gen3"
]

# Trova file una volta sola
vcf_files = []
for folder in vcf_folders:
    for chr_num in regions_by_chr:
        vcf_files.extend(glob.glob(os.path.join(folder, f"*chr{chr_num}.vcf.gz")))

vcf_files = list(set(vcf_files))  # evita duplicati

all_variants = {}
samples_order = []

for vcf_file in vcf_files:
    print("Leggo", vcf_file)
    v = VCF(vcf_file)
    samples = v.samples
    # aggiorna ordine globale
    for s in samples:
        if s not in samples_order:
            samples_order.append(s)

    for rec in v:
        chr_num = rec.CHROM
        if chr_num not in regions_by_chr:
            continue

        hits = regions_by_chr[chr_num][rec.POS]
        if not hits:
            continue  # non in nessun variant richiesto

        for hit in hits:  # possibile variante cade in più geni
            variant = hit.data
            variant_id = f"{variant}:{rec.CHROM}:{rec.POS}:{rec.REF}:{rec.ALT[0]}"

            gt = rec.gt_types  # array di 0,1,2,3
            genotypes = {s: (g if g != 3 else -1) for s, g in zip(samples, gt)}

            if variant_id in all_variants:
                all_variants[variant_id].update(genotypes)
            else:
                all_variants[variant_id] = genotypes

# ---- Costruzione DataFrame ----
df = pd.DataFrame.from_dict(all_variants, orient="columns")
df = df.reindex(samples_order).fillna(-1).astype(int)

df = df.loc[:, (df == -1).mean() < 0.30]


out = f"/tmp/variants_chr_{'_'.join(genes_input)}.csv"
df.index.name = "id"
df.to_csv(out, sep=";")
print(f"Salvato {out} ({df.shape[0]} samples, {df.shape[1]} varianti)")
