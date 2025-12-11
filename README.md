# Gene × Environment Interaction Analysis

Analisi di interazioni **gene × ambiente** basata su dati genetici e esposizioni ambientali.
Il progetto esegue matching dei controlli, fit di modelli lineari e test di permutazione per valutare l'effetto delle interazioni sul **tempo di insorgenza** (`onset_age`).

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Struttura del progetto

- `main.py` - script principale con analisi e visualizzazione.
- `db.py` - funzioni per salvare e caricare risultati.
- `gen_diminuito.csv` - dati genetici.
- `componenti_ambientali.csv` - esposizioni ambientali e covariate.
- `temp_df.pkl` - file temporaneo per processi paralleli.

---

## Requisiti

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib tqdm
