import gzip
from typing import Dict, List, Set

class CTDAPI:

    GENE_DISEASE_FILE = "CTD_curated_genes_diseases.tsv.gz"
    CHEM_GENE_FILE = "CTD_chem_gene_ixns.tsv.gz"

    @staticmethod
    def _get_header_and_rows(f):
        header = None
        rows = []
        for line in f:
            line = line.strip()
            if line.startswith("# Fields:"):
                # la riga subito dopo è l'header
                header_line = next(f).strip()
                header = [h.strip() for h in header_line.lstrip("#").split("\t")]
                break
        if header is None:
            raise ValueError("Header # Fields: non trovato")
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.strip().split("\t"))
        return header, rows

    @staticmethod
    def query_gene(gene_symbol: str) -> Dict[str, List[str]]:
        chemicals: Set[str] = set()
        neuro_diseases: Set[str] = set()

        # Interazioni chemical-gene
        try:
            with gzip.open(CTDAPI.CHEM_GENE_FILE, mode="rt", encoding="utf-8") as f:
                header, rows = CTDAPI._get_header_and_rows(f)
                for r in rows:
                    row = dict(zip(header, r))
                    if row.get("GeneSymbol", "").upper() == gene_symbol.upper():
                        chem = row.get("ChemicalName")
                        if chem:
                            chemicals.add(chem.strip())
        except FileNotFoundError:
            print(f"Errore: file {CTDAPI.CHEM_GENE_FILE} non trovato.")
        except Exception as e:
            print(f"Errore durante la lettura di {CTDAPI.CHEM_GENE_FILE}: {e}")

        # Malattie neurologiche curate
        try:
            with gzip.open(CTDAPI.GENE_DISEASE_FILE, mode="rt", encoding="utf-8") as f:
                header, rows = CTDAPI._get_header_and_rows(f)
                for r in rows:
                    row = dict(zip(header, r))
                    if row.get("GeneSymbol", "").upper() == gene_symbol.upper():
                        disease = row.get("DiseaseName")
                        if disease and "neuro" in disease.lower():
                            neuro_diseases.add(disease.strip())
        except FileNotFoundError:
            print(f"Errore: file {CTDAPI.GENE_DISEASE_FILE} non trovato.")
        except Exception as e:
            print(f"Errore durante la lettura di {CTDAPI.GENE_DISEASE_FILE}: {e}")

        return {
            "chemicals": list(chemicals),
            "neuro_diseases": list(neuro_diseases)
        }
