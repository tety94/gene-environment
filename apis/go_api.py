import requests
from config import GO_REST
from apis.ensembl_api import EnsemblAPI

#GENE ONTOLOGY API
class GOAPI:

    @staticmethod
    def get_go_terms(uniprot: str) -> dict:

        url = f"{GO_REST}/annotation/search"
        params = {
            "geneProductId": uniprot,
            "limit": 100
        }

        r = requests.get(url, params=params)

        if not r.ok:
            return {"neuro": [], "toxic": []}

        neuro = set()
        toxic = set()

        for res in r.json().get("results", []):
            go_name = res["goName"]
            if go_name:
                term = go_name.lower()
                if any(x in term for x in ["neuron", "synapse", "axon"]):
                    neuro.add(res["goName"])
                if any(x in term for x in ["toxic", "oxidative", "stress"]):
                    toxic.add(res["goName"])

        return {
            "neuro": list(neuro),
            "toxic": list(toxic)
        }
