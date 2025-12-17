import requests
from config import GTEX_REST

class GTExAPI:

    @staticmethod
    def get_brain_expression(gene_symbol: str):
        """
        Restituisce un dizionario con:
        - expressed_brain: True/False
        - tissues: lista di tessuti cerebrali
        """
        url = f"{GTEX_REST}/expression/geneExpression"
        params = {
            "gencodeId": gene_symbol,  # o usare geneSymbol, da verificare
            "format": "json"
        }
        r = requests.get(url, params=params)

        if not r.ok:
            return {"expressed_brain": False, "tissues": []}

        data = r.json().get("geneExpression", [])
        brain_tissues = [d["tissueSiteDetail"] for d in data if "Brain" in d["tissueSiteDetail"]]

        return {
            "expressed_brain": len(brain_tissues) > 0,
            "tissues": brain_tissues
        }
