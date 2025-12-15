import requests

def fetch_gene(chrom, pos):
    url = f"https://rest.ensembl.org/overlap/region/human/{chrom}:{pos}-{pos}?feature=gene;content-type=application/json"
    r = requests.get(url, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    data = r.json()
    if data:
        try:
            return data[0]['external_name']  # prende il primo gene trovato
        except:
            print(data)
    return None
