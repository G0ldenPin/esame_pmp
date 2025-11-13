# esame_pmp
Progetto A.A. 2025/2026 per Principi e Modelli della Percezione in Università degli Studi di Milano

# 🍄 Progetto di Machine Learning: Classificazione dei Funghi (Mushroom Dataset)

## 🌟 Panoramica del Progetto

Questo progetto è stato sviluppato come parte dell'esame di **Principi e Modelli della Percezione** ed è un'analisi collaborativa incentrata sulla classificazione del dataset **Mushroom** proveniente dall'archivio UCI.

L'obiettivo principale è sviluppare un modello di Machine Learning (ML) robusto, in grado di predire con alta affidabilità se un fungo è **commestibile** (*edible*) o **velenoso** (*poisonous*), basandosi sulle sue 22 caratteristiche categoriali.

---

## 💾 Dataset

Il dataset di riferimento è l'**UCI Mushroom Dataset** (fonte: [https://archive.ics.uci.edu/dataset/73/mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)).

* **Fonte:** UCI Machine Learning Repository
* **Target:** La variabile di classe (prima colonna) indica se il fungo è **'e'** (edible/commestibile) o **'p'** (poisonous/velenoso).
* **Caratteristiche:** 22 attributi (tutti categoriali) che descrivono aspetti come forma del cappello, colore, odore, ecc.

---

## 🛠️ Metodologia e Tecniche Utilizzate

### 1. Exploratory Data Analysis (EDA)
È stata eseguita un'analisi esplorativa per comprendere la distribuzione delle classi e delle singole caratteristiche, identificando le variabili più correlate con la tossicità.

### 2. Pre-elaborazione dei Dati
Dato che il dataset è composto interamente da variabili categoriali, è stata necessaria una trasformazione:
* **Tecnica:** **[Specificare la tecnica usata, es.: One-Hot Encoding]** è stata applicata per convertire gli attributi in un formato numerico (0 e 1) adatto ai modelli di ML.
* **Suddivisione:** I dati sono stati suddivisi in set di addestramento e di test.

### 3. Modellazione
È stato implementato un modello di Classificazione per predire la classe target.

* **Tecnica ML Scelta:** **[Inserire il modello esatto, es.: Random Forest Classifier, Rete Neurale Multistrato]**
* **Libreria:** Utilizzo della libreria **[Specificare la libreria, es.: Scikit-learn, TensorFlow/Keras]**.
* **Valutazione:** Le performance del modello sono state valutate utilizzando le metriche **[Specificare le metriche, es.: Accuracy, Precision, Recall, Matrice di Confusione]**.

---

## 📊 Risultati Ottenuti

Il modello **[Nome del Modello]** ha dimostrato **[Inserire una breve descrizione del risultato, es.: un'elevatissima capacità predittiva]**, ottenendo i seguenti risultati sul *test set*:

* **Accuracy:** **[Inserire la percentuale di Accuracy, es.: 99.98%]**
* **[Inserire un'altra metrica chiave, es.: F1-Score]:** **[Inserire il valore]**

**[Aggiungere una breve conclusione sui risultati, es.: La classificazione ha dimostrato che il modello è quasi perfetto nel distinguere i funghi velenosi da quelli commestibili.]**

---

## 📂 Contenuti del Repository

Questo repository contiene i seguenti file principali:

* `[Nome del Notebook/Script principale, es.: mushroom_classification.ipynb]`: Il codice Python completo che include EDA, pre-elaborazione, addestramento del modello e valutazione.
* `requirements.txt`: Elenco delle librerie Python necessarie per eseguire il progetto.



---

## 🧑‍💻 Team e Contributi

Questo progetto è frutto della collaborazione tra i seguenti membri:

| Nome e Cognome | Contributo Principale |
| :--- | :--- |
| **Lisa G. Bassetti** | [Specificare il ruolo, es.: Sviluppo del modello di Classificazione e Documentazione del codice] |
| **Matteo COtugno** | [Specificare il ruolo, es.: Acquisizione dati, Pre-elaborazione (One-Hot Encoding) e EDA (Analisi Esplorativa)] |
