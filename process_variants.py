from db import get_conn, update_variant_gene, get_empty_variants_gene
from ensembl_api import fetch_gene


def main():
    conn = get_conn()
    variants = get_empty_variants_gene(conn)

    for variant in variants:
        chrom, pos = variant['chromosome'], variant['position']
        gene = fetch_gene(chrom, pos)
        if gene:
            update_variant_gene(conn, variant, gene)
            print(f"{variant} aggiornato con gene {gene}")
        else:
            print(f"Nessun gene trovato per {variant}")

    conn.close()


if __name__ == "__main__":
    main()
