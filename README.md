# Gene × Environment Interaction Analysis

Analisi di interazioni **gene × ambiente** basata su dati genetici e esposizioni ambientali.
Lo script principale utilizza **matching dei controlli**, **modelli lineari** e **test di permutazione** per valutare l'effetto delle interazioni sul **tempo di insorgenza** (`onset_age`).

---

## Struttura del progetto

* `main.py` - script principale con pipeline completa: preprocessing, matching, modellazione, permutazione e visualizzazione.
* `db.py` - funzioni per salvare e caricare i risultati dei geni.
* `gen_diminuito.csv` - dati genetici (genotipi dei pazienti).
* `componenti_ambientali.csv` - esposizioni ambientali e covariate.
* `temp_df.pkl` - file temporaneo condiviso tra processi paralleli.

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
onset_col = "onset_age"                   
exposures = ["seminativi_1500"]           
covariates = ["sex", "onset_site", "diagnostic_delay"]
n_perm = 1000                             
random_state = 42
standardize = True
min_treated = 5
min_sample_size = 10
max_workers = 8                           
match_k = 3                               
```

---

## Funzionamento di `main.py`

Lo script `main.py` gestisce l’intera pipeline di analisi delle interazioni **gene × ambiente**.

### 1. Caricamento dei dati

* Carica il file genetico (`gen_diminuito.csv`) e trasforma i geni in variabili binarie (`0`/`1`).
* Carica il file delle esposizioni ambientali e covariate (`componenti_ambientali.csv`).
* Esegue il merge dei due dataset usando la colonna `id`.
* Converte la colonna `onset_age` in numerico, gestendo eventuali valori non validi.

### 2. Standardizzazione delle esposizioni

* Se `standardize=True`, le variabili ambientali vengono scalate con **StandardScaler** (media 0, deviazione standard 1).
* Le variabili standardizzate vengono aggiunte al dataframe con suffisso `_std`.

### 3. Ridenominazione dei geni

* Rinomina le colonne dei geni in nomi sicuri (`gene_0`, `gene_1`, …) per evitare conflitti nei processi paralleli.
* Salva il mapping dei nomi originali → nomi sicuri.
* Salva il dataframe completo in `temp_df.pkl` per il riutilizzo nei processi paralleli.

### 4. Matching dei controlli

* Per ogni gene, seleziona unità trattate (`gene==1`) e controlli (`gene==0`).
* Esegue **nearest-neighbor matching** basato su covariate numeriche e categoriche.
* Supporta caching del matching (`matched_{gene}.pkl`) per velocizzare permutazioni ripetute.

### 5. Fit dei modelli lineari

* Modello lineare per ciascun gene:

```
onset_age ~ gene * (exposures) + covariates
```

* Estrae il coefficiente dell’interazione gene × esposizione.

### 6. Test di permutazione

* Permuta la variabile gene sull’intero dataset originale.
* Ricrea il matching e rifit il modello per ogni permutazione.
* Costruisce la distribuzione nulla dei beta permutati.
* Calcola il **p-value empirico** come proporzione di beta permutati più estremi di quello osservato.

### 7. Parallelizzazione

* Analisi parallela di tutti i geni usando `ProcessPoolExecutor`.
* Ogni processo chiama `process_single_gene()` per gestire matching, fit, permutazioni e salvataggio dei risultati.

### 8. Correzione multipla

* Calcola la **FDR** (False Discovery Rate) sui p-value empirici usando il metodo di **Benjamini-Hochberg**.

### 9. Visualizzazione dei risultati

* Genera un **volcano plot** (`volcano_plot.png`) con:

  * Asse X: coefficiente dell’interazione (`beta`)
  * Asse Y: -log10(p-value empirico)
  * Evidenzia i geni significativi dopo FDR (`fdr < 0.05`)

---

## Funzioni chiave in `main.py`

* `build_formula()` → costruisce la formula del modello lineare.
* `_prepare_matching_matrix()` → prepara la matrice numerica per il matching (gestione variabili categoriche).
* `match_control_units()` → esegue il nearest-neighbor matching tra trattati e controlli.
* `process_single_gene()` → pipeline completa per un singolo gene: matching, fit, permutazione, salvataggio.
* `permutation_test_interaction()` → esegue il test di permutazione per l’interazione gene × ambiente.
* `add_fdr()` → calcola la FDR sui p-value empirici.
* `volcano_plot()` → genera il volcano plot.
* `load_or_compute_matched()` → carica o calcola il matching con caching.

---

## Esecuzione

```bash
python main.py
```

Output generati:

* Risultati dei geni salvati tramite `db.py`.
* Volcano plot in `volcano_plot.png`.

---

## Note

* Il matching e il test di permutazione possono essere intensivi: regola `max_workers` in base alla CPU disponibile.
* File temporanei `matched_{gene}.pkl` vengono rimossi automaticamente dopo l’uso.

---

## Licenza

MIT License
