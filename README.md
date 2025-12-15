# Gene × Environment Interaction Analysis

Analisi di interazioni **variant × ambiente** basata su dati varianttici e esposizioni ambientali.
La pipeline principale esegue **matching dei controlli**, **modellazione lineare** e **test di permutazione** per valutare l’effetto delle interazioni sul **tempo di insorgenza** (`onset_age`).

---

## Struttura del progetto

- `main.py` - script principale con pipeline completa: preprocessing, matching, modellazione, permutazione e visualizzazione.
- `db.py` - funzioni per salvare e caricare i risultati dei geni.
- `gen_diminuito.csv` - dati varianttici (genotipi dei pazienti).
- `componenti_ambientali.csv` - esposizioni ambientali e covariate.
- `temp_df.pkl` - file temporaneo condiviso tra processi paralleli.

---

## Requisiti

Python 3.9+ con le seguenti librerie:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib tqdm
```

---

## Configurazione

All’inizio di `main.py` puoi modificare:

```python
raw_file = "gen_diminuito.csv"
env_file = "componenti_ambientali.csv"
sep = ';'
decimal = '.'
onset_col = "onset_age"
exposures = ["seminativi_1500"]
covariates = ["sex", "onset_site", "diagnostic_delay"]
n_perm = 1000              # numero di permutazioni
random_state = 42
standardize = True          # se True, le esposizioni vengono standardizzate
min_treated = 5             # minimo trattati per fare il matching
min_sample_size = 10        # minimo campione per il modello
max_workers = 16            # parallelizzazione
TEMP_DF_PATH = "temp_df.pkl"
match_k = 3                 # numero controlli per ciascun trattato
```

---

## Funzionamento di `main.py`

### 1. Caricamento dati

- Dati varianttici (`gen_diminuito.csv`) filtrando colonne con troppi `-1` e convertendo in variabili binarie (`0`/`1`).
- Dati ambientali e covariate (`componenti_ambientali.csv`), con `sex` e `onset_site` come categorie.
- Merge dei due dataset tramite colonna `id`.
- Conversione di `onset_age` in numerico.

### 2. Standardizzazione delle esposizioni

- Se `standardize=True`, le variabili ambientali sono scalate (`mean=0, std=1`) e aggiunte con suffisso `_std`.

### 3. Nomi sicuri dei geni

- Le colonne dei geni sono rinominate (`variant_0`, `variant_1`, …) per evitare conflitti nei processi paralleli.
- Mapping originale ↔ nome sicuro salvato in memoria.
- Salvataggio del dataframe in `temp_df.pkl` per i processi paralleli.

### 4. Matching dei controlli

- Nearest-neighbor matching per ciascun variant (trattati `variant==1`, controlli `variant==0`) basato su covariate numeriche e categoriche.
- Diagnostica tramite **Standardized Mean Differences (SMD)**.
- Possibilità di caching dei matching (`matched_{variant}.pkl`).

### 5. Fit dei modelli lineari

- Modello lineare per ciascun variant:

```
onset_age ~ variant * (exposures) + covariates
```

- Estrazione del coefficiente dell’interazione variant × esposizione.

### 6. Test di permutazione

- Permuta la variabile variant sull’intero dataset originale.
- Ricrea il matching e rifit il modello per ciascuna permutazione.
- Costruisce la distribuzione nulla dei beta permutati.
- Calcola il **p-value empirico** come proporzione dei beta permutati più estremi di quello osservato.

### 7. Parallelizzazione

- Analisi parallela di tutti i geni con `ProcessPoolExecutor`.
- Ogni processo esegue `process_single_variant()` per matching, fit, permutazioni e salvataggio.

### 8. Correzione multipla

- Calcolo della **FDR** (False Discovery Rate) sui p-value empirici usando il metodo **Benjamini-Hochberg**.

### 9. Visualizzazione dei risultati

- Generazione di un **volcano plot** (`volcano_plot.png`) con:

  - Asse X: coefficiente dell’interazione (`beta`)
  - Asse Y: -log10(p-value empirico)
  - Evidenziazione dei geni significativi dopo FDR (`fdr < 0.05`)

---

## Funzioni chiave in `main.py`

- `build_formula()` → costruisce la formula del modello lineare.
- `_prepare_matching_matrix()` → prepara la matrice numerica per il matching (gestione variabili categoriche e missing).
- `match_control_units()` → esegue il nearest-neighbor matching.
- `check_balance()` → calcola SMD per diagnosticare il bilanciamento delle covariate.
- `process_single_variant()` → pipeline completa per un singolo variant: matching, fit, permutazione, salvataggio.
- `add_fdr()` → calcola FDR sui p-value empirici.
- `volcano_plot()` → variantra il volcano plot.
- `load_or_compute_matched()` → carica o calcola il matching con caching opzionale.

---

## Esecuzione

```bash
python main.py
```

Output variantrati:

- Risultati dei geni salvati tramite `db.py`.
- Volcano plot in `volcano_plot.png`.

---

## Note

- Matching e test di permutazione possono essere intensivi: regola `max_workers` in base alla CPU disponibile.
- File temporanei `matched_{variant}.pkl` possono essere rimossi automaticamente.
- SMD > 0.25 indica possibili squilibri nel matching.

---

## Licenza

MIT License
