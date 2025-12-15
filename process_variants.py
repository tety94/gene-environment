from db import get_conn, update_variant_gene, get_empty_variants_gene
from ensembl_api import fetch_gene


def main():
    conn = get_conn()
    variants = get_empty_variants_gene()

    for _, variant in variants.iterrows():
        chrom, pos = variant['chromosome'], variant['position']
        gene_id, gene_name = fetch_gene(chrom, pos)
        if gene_id:
            update_variant_gene(conn, variant['variant'], gene_id,gene_name)
            print(f"{variant['variant']} aggiornato con gene {gene_id}")
        else:
            update_variant_gene(conn, variant['variant'], 'NO-GENE','NO-GENE')
            print(f"Nessun gene trovato per {variant}")

    conn.close()


if __name__ == "__main__":
    main()
