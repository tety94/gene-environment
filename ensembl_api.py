import requests

def fetch_gene(chrom, pos):
    url = f"https://rest.ensembl.org/overlap/region/human/{chrom}:{pos}-{pos}?feature=gene;content-type=application/json"
    r = requests.get(url, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    data = r.json()
    if data:
        gene = data[0]
        gene_id = gene['id']
        gene_name = gene.get('external_name', gene_id)
        return gene_id, gene_name
    return None, None
