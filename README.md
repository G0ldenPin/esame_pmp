# 🍄 Visualizzatore Interattivo del Dataset Mushroom
Progetto A.A. 2025/2026 – Principi e Modelli della Percezione
Università degli Studi di Milano

## 🌟 Panoramica del Progetto

L’obiettivo di questo progetto è analizzare il dataset Mushroom dell’UCI Machine Learning Repository e costruire un modello di classificazione in grado di prevedere se un fungo è edibile (e) o velenoso (p) basandosi su caratteristiche osservabili a vista. Il progetto si articola in tre parti:

1.  **Caricamento e preparazione dei dati**: Il dataset originale viene caricato e preparato per l'analisi.
2.  **Addestramento del modello**: Viene addestrato un classificatore (Decision Tree) per distinguere i funghi velenosi da quelli commestibili.
3.  **Interfaccia utente (GUI)**: È stata sviluppata un'applicazione grafica che permette di interrogare il modello, inserendo le caratteristiche di un fungo per ottenerne una predizione sulla commestibilità.

Tutto il codice è implementato in Python, utilizzando librerie scientifiche standard come Pandas e Scikit-learn.

---

## 💾 Dataset

Il dataset di riferimento è l'**UCI Mushroom Dataset** (fonte: [https://archive.ics.uci.edu/dataset/73/mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)).

*   **Fonte:** UCI Machine Learning Repository
*   **Target:** La variabile di classe (prima colonna) indica se il fungo è **'e'** (edible/commestibile) o **'p'** (poisonous/velenoso).
*   **Caratteristiche:** 22 attributi (tutti categoriali) che descrivono aspetti come forma del cappello, colore, odore, ecc. Per questo progetto, sono state selezionate 5 feature principali per l'addestramento.

---

## 📦 Istruzioni per l'utilizzo

Per il setup del progetto consigliamo di usare PyCharm come IDE per la sua semplicità nell'installazione delle dipendenze.

1.  **Copia la repo**:
    ```bash
    git clone https://github.com/G0ldenPin/esame_pmp
    ```
2.  **Apri il progetto e installa le dipendenze** in PyCharm:
    ![Pycharm lib installer](md_images/pyinstaller.png)
    
    **N.B.** una lista completa delle dipendenze è disponibile alla fine di questo tutorial.
3.  **Prepara i dati** (da eseguire solo la prima volta):
    Questo comando scarica il dataset e lo distribuisce nelle cartelle corrette del progetto.
    ```bash
    python3 data/setup_dataset.py
    ```
    In caso di problemi di rete (es. errori SSL), lo script tenterà di usare un metodo di download alternativo. Se anche questo fallisce, puoi usare il metodo manuale:
    1. Scarica il dataset da [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification) o un'altra fonte e salvalo come file ZIP.
    2. Salva il file come `mushroom.zip` nella cartella `data/`.
    3. Esegui il comando seguente per estrarre i dati dallo ZIP:
    ```bash
    python3 data/setup_dataset.py --zip
    ```
4.  **Addestra il modello e visualizza l'albero**:
    Questo comando ri-addestra il modello e genera l'immagine `decision_tree.png`.
    ```bash
    python poison_model.py
    ```
5.  **Avvia la GUI per testare il modello**:
    ```bash
    python poison_tester_gui.py
    ```
6.  **Genera statistiche di valutazione**:
    Questo comando crea il file `report_statistiche.txt`.
    ```bash
    python test_stats.py
    ```
7. **Visualizza i dati** che ti servono tramite grafici:
   ```bash
    python funghiPrototipo.py
    ```
   
**LISTA DELLE DIPENDENZE** :

* **pandas**: Libreria per la manipolazione e l'analisi di dati, usata per gestire il dataset dei funghi in tabelle.
* **matplotlib**: Libreria per la creazione di grafici e visualizzazioni.
* **seaborn**: Basata su matplotlib, semplifica la creazione di grafici statistici esteticamente gradevoli.
* **ucimlrepo**: Libreria specifica per scaricare dataset dall'UCI Machine Learning Repository.
* **tkinter**: La libreria standard di Python per creare interfacce grafiche (GUI).
* **scikit-learn**: Fornisce l'algoritmo `DecisionTreeClassifier` e strumenti per la preparazione dei dati e la valutazione del modello.
* **requests**: Libreria per effettuare richieste HTTP.
* **io**: Modulo standard di Python per gestire flussi di dati.

---

## 🛠️ Metodologia e Tecniche Utilizzate

Il cuore del progetto è un `DecisionTreeClassifier` dalla libreria `scikit-learn`.

1.  **Preparazione Dati**: Le feature testuali (es. "Marrone", "Convesso") vengono convertite in valori numerici tramite `LabelEncoder`, poiché i modelli di machine learning operano su dati numerici.
2.  **Addestramento**: Il modello viene addestrato (`fit`) utilizzando un sottoinsieme di feature selezionate (`cap-shape`, `cap-color`, `gill-color`, `stalk-shape`, `odor`) e la variabile target ('class'). L'algoritmo impara a creare una struttura ad albero basata su domande (es. "L'odore è pungente?"), ottimizzata per separare i funghi commestibili da quelli velenosi.
3.  **Predizione**: L'interfaccia grafica raccoglie gli input dell'utente, li codifica numericamente usando gli stessi `LabelEncoder` dell'addestramento e li passa al modello per ottenere una predizione ('e' o 'p').

---

## 🌳 Visualizzazione dell’Albero Decisionale

Il progetto include la possibilità di visualizzare l'albero decisionale per comprendere come il modello prende le sue decisioni. Eseguendo lo script `poison_model.py`, viene generata e salvata un'immagine (`decision_tree.png`) che mostra l'albero completo.

La feature più importante, scelta come radice dell'albero, è quasi sempre l'**odore (`odor`)**, poiché è il singolo attributo più informativo per determinare la velenosità di un fungo in questo dataset.

**Esempio di regole osservate dall'albero**:
*   Se `odor` = `pungent` (p) O `fetid` (f) -> **Velenoso**
*   Se `odor` = `none` (n) E `gill-color` = `buff` (b) -> **Velenoso**
*   Se `odor` = `almond` (a) O `anise` (l) -> **Commestibile**

Queste regole mostrano come il modello naviga tra le caratteristiche per arrivare a una classificazione.

---

## 📊 Risultati Ottenuti

Il modello `DecisionTreeClassifier` addestrato sul dataset completo raggiunge un'accuratezza molto elevata, di circa il 90% (all'ultimo test eseguito prima della consegna), nel classificare correttamente i funghi del training set. Questo indica che le feature selezionate sono sufficientemente potenti da creare regole di separazione molto precise.

L'interfaccia grafica, permette a chiunque di sfruttare la potenza del modello senza alcuna conoscenza di programmazione, fornendo un'esperienza utente semplice e immediata per la classificazione.

Per visualizzare localmente le statistiche dei risultati del modello si può utilizzare il file  `test_stats.py`

---

## 📂 Contenuti del Repository

*   `data/`: Cartella per la gestione dei dati.
    *   `setup_dataset.py`: Script per scaricare/estrarre il dataset e distribuirlo nel progetto. **DA ESEGUIRE LA PRIMA VOLTA.**
*   `poison_analysis/`: Cartella contenente il modello di ML e i relativi script di funzionamento.
    *   `poison_model.py`: Script che definisce, addestra e gestisce il classificatore `MushroomClassifier`.
    *   `poison_tester_gui.py`: L'applicazione con interfaccia grafica (basata su Tkinter) per testare il modello.
    *   `test_stats.py`: Script per generare un file testuale con le statistiche del modello di ML e la confusion matrix del modello.
*   `visualization/`: Cartella contenente tutto il necessario per la visualizzazione grafica del dataframe.
    *   `VisualKmodes.py`: Script specifico per visualizzare i risultati dell'algoritmo di clustering K-Modes.
    *   `VisualKmodesCSV.py`: Script specifico per visualizzare i risultati dell'algoritmo di clustering K-Modes utilizzando il file CSV del dataset.
*   `README.md`: Questo file.

---

## 🧑‍💻 Team e Contributi

Questo progetto è frutto della collaborazione tra i seguenti membri:

| Nome e Cognome     | Contributo Principale           |
| :----------------- |:--------------------------------|
| **Lisa G. Bassetti** | Poison analysis e decision tree |
| **Matteo Cotugno**   | Data visualization e Clustering |
