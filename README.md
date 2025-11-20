# 🍄 Visualizzatore Interattivo del Dataset Mushroom
Progetto A.A. 2025/2026 – Principi e Modelli della Percezione
Università degli Studi di Milano

## 🌟 Panoramica del Progetto

L’obiettivo di questo progetto è analizzare il dataset Mushroom dell’UCI Machine Learning Repository e costruire un modello di classificazione in grado di prevedere se un fungo è edibile (e) o velenoso (p) basandosi su caratteristiche osservabili a vista. Il progetto si articola in tre parti:

1. Caricamento e pulizia dei dati
2. Visualizzazione e analisi esplorativa
3. Modello di classificazione (Decision Tree + altri algoritmi)

Tutto il codice è implementato in Python, utilizzando librerie scientifiche standard e import vari per migliorare la qualità dell'output.

---

## 💾 Dataset

Il dataset di riferimento è l'**UCI Mushroom Dataset** (fonte: [https://archive.ics.uci.edu/dataset/73/mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)).

* **Fonte:** UCI Machine Learning Repository
* **Target:** La variabile di classe (prima colonna) indica se il fungo è **'e'** (edible/commestibile) o **'p'** (poisonous/velenoso).
* **Caratteristiche:** 22 attributi (tutti categoriali) che descrivono aspetti come forma del cappello, colore, odore, ecc.

---

## 📦 Istruzioni per l'utilizzo

Il metodo d'installazione che consigliamo per il progetto utilizza uv e venv e si articola nei seguenti passaggi:
1. Installa venv tramite python
2. Installa uv tramite pip
   ```
   pip install uv
   ```
   N.B. Se non è già installato, installare anche pip.
3. Crea l'ambiente virtuale usando uv
   ```
   uv venv
   source .venv/bin/activate
   ```
4. Installare le dipendenze
   ```
   uv pip install pandas seaborn matplotlib scikit-learn
   OPPURE
   pip install pandas seaborn matplotlib scikit-learn
   ```
**PER VISUALIZZARE I DATI**
 1. Nella cartella *visualization*, utilizzare il file *funghi.py*
    comparirà una GUI dedicata alla visualizzazione dei grafici.
 2. Utilizzare la GUI per scegliere che grafico visualizzare.

**PER UTILIZZARE L'ANALIZZATORE DI FUNGHI**
1. Nella cartella *poison_analysis* utilizzare il file *data_setup.py* per generare il file csv utilizzato dal modello.
2. Utilizzare il file *poison_tester_gui.py* per utilizzare la GUI dedicata al tool.
3. Per analizzare le prestazioni del modello utilizzare il file *test_stats.py*, che genererà un report dettagliato sulle analisi dell'accuratezza. 

---

## 🛠️ Metodologia e Tecniche Utilizzate
**[WIP]**

---

🌳 Albero Decisionale

Il progetto include:

**[WIP]**

Esempio di regole osservate:
**[WIP]**

---

## 📊 Risultati Ottenuti

I seguenti risultati sono ottenuti dall'ultima run del codice eseguita prima della consegna del progetto il 26/11/25.

=============================================
REPORT STATISTICO COMPLETO - MUSHROOM DATASET
=============================================

1. ACCURATEZZA (Cross-Validation 5-fold)
----------------------------------------
Media: 90.69%
Deviazione Standard: +/- 10.36%

2. REPORT DI CLASSIFICAZIONE
----------------------------------------
              precision    recall  f1-score   support

Commestibile       0.99      1.00      0.99      1263
    Velenoso       1.00      0.99      0.99      1175

    accuracy                           0.99      2438
   macro avg       0.99      0.99      0.99      2438
weighted avg       0.99      0.99      0.99      2438


3. DETTAGLIO ERRORI (Matrice di Confusione)
----------------------------------------
Su un totale di 2438 funghi testati:
- Veri Negativi (Commestibili corretti): 1258
- Veri Positivi (Velenosi corretti):     1165
- Falsi Positivi (Allarmi inutili):      5
- Falsi Negativi (PERICOLOSI):           10 <--- Questo numero deve essere 0!

---

## 📂 Contenuti del Repository

**[WIP]**


---

## 🧑‍💻 Team e Contributi

Questo progetto è frutto della collaborazione tra i seguenti membri:

| Nome e Cognome | Contributo Principale |
| :--- | :--- |
| **Lisa G. Bassetti** | Decision Tree |
| **Matteo Cotugno** | Data Visualization |
