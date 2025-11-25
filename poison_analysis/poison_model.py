import pandas as pd
import numpy as np

# Da 'scikit-learn', una delle più importanti librerie di machine learning, importiamo
# il 'DecisionTreeClassifier'. Questo è l'algoritmo che useremo per costruire il nostro
# modello di predizione, simile a un diagramma di flusso che impara dai dati.
from sklearn.tree import DecisionTreeClassifier

# Importiamo anche il 'LabelEncoder', un'utilità per convertire le etichette
# testuali (es. "marrone", "bianco") in numeri, poiché i modelli di machine learning
# lavorano solo con dati numerici.
from sklearn.preprocessing import LabelEncoder

# Infine, importiamo 'os' per interagire con il sistema operativo. Ci serve per
# gestire i percorsi dei file in modo che lo script funzioni su computer diversi.
import os
import matplotlib.pyplot as plt
from sklearn import tree

class MushroomClassifier:

    def __init__(self, csv_path='mushrooms.csv'):

        # --- Inizializzazione delle Proprietà ---

        # init a 'None', perché verrà creato e addestrato nel metodo '_train'.
        self.model = None

        # dict di labelencoders che trasforma i dati testuali in numeri per il training ma anche per le nuove predizioni.
        self.dizionario_encoders = {}

        # encoder specifico per il target
        self.le_target = None

        # --- Configurazione delle Feature ---

        # lista delle feature scelte
        self.features_input = [
            'cap-shape',
            'cap-color',
            'gill-color',
            'stalk-shape',
            'odor'
        ]

        # dict che mappa le feature italiane in singole lettere
        self.menu_opzioni = {
            "cap-shape": {"A campana": "b", "Conico": "c", "Convesso": "x", "Piatto": "f", "Nodoso/Umbone": "k", "Infossato": "s"},
            "cap-color": {"Marrone": "n", "Giallo pallido (Buff)": "b", "Cannella": "c", "Grigio": "g", "Verde": "r", "Rosa": "p", "Viola": "u", "Rosso": "e", "Bianco": "w", "Giallo": "y"},
            "gill-color": {"Nero": "k", "Marrone": "n", "Giallo pallido": "b", "Cioccolato": "h", "Grigio": "g", "Verde": "r", "Arancione": "o", "Rosa": "p", "Viola": "u", "Rosso": "e", "Bianco": "w", "Giallo": "y"},
            "stalk-shape": {"Si allarga alla base": "e", "Si restringe (affusolato)": "t"},
            "odor": {"Mandorla": "a", "Anice": "l", "Chimico/Creosoto": "c", "Pesce": "y", "Fetido": "f", "Muffa": "m", "Nessun odore": "n", "Pungente": "p", "Speziato": "s"}
        }

        # dict che mappa i nomi delle feature inglesi a quelli italiani
        self.nomi_features_ita = {
            'cap-shape': 'FORMA DEL CAPPELLO', 'cap-color': 'COLORE DEL CAPPELLO', 'gill-color': 'COLORE DELLE LAMELLE',
            'stalk-shape': 'FORMA DEL GAMBO', 'odor': "ODORE"
        }

        # --- Caricamento Dati e Addestramento ---

        #check per controllare il path assoluto del file csv se il file non viene trovato così com'è
        script_dir = os.path.dirname(__file__)
        full_path = os.path.join(script_dir, os.path.basename(csv_path))

        if not os.path.exists(full_path):
             full_path = csv_path

        try:
            self._train(full_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"ERRORE: Il file '{os.path.basename(csv_path)}' non è stato trovato. Assicurati che sia nella stessa cartella dell'eseguibile.")

    def _train(self, csv_path):

        df = pd.read_csv(csv_path)

        # Separiamo le feature (colonne input -> X) dalla variabile target (la predizione -> y).
        X = df[self.features_input].copy()  # .copy() previene un warning di pandas
        y = df['poisonous']

        # trasformiamo ogni feature testuale in numeri.
        self.dizionario_encoders = {col: LabelEncoder() for col in X.columns}
        for col, encoder in self.dizionario_encoders.items():
            # 'fit_transform' impara le etichette uniche e le trasforma in sequenze numeriche
            X[col] = encoder.fit_transform(X[col])

        self.le_target = LabelEncoder()
        y = self.le_target.fit_transform(y) # 'e' e 'p' diventeranno 0 e 1

        # Creiamo un'istanza del decision tree.
        # #'random_state=42' serve a garantire che l'addestramento dia sempre lo stesso risultato, rendendo il modello riproducibile.
        self.model = DecisionTreeClassifier(random_state=42)

        # 'self.model.fit' : il modello analizza le feature (X) e i risultati (y) per imparare le regole di classificazione.
        self.model.fit(X, y)

    def predict(self, features_dict):

        # Controllo di sicurezza per assicurarsi di aver ricevuto il numero corretto di feature.
        if len(features_dict) != len(self.features_input):
            raise ValueError(f"L'input deve contenere {len(self.features_input)} feature.")

        encoded_input = []

        # iteriamo sulla lista 'self.features_input' per garantire che l'ordine delle feature sia corretto.
        # prendiamo l'input utente (encoded) e ne recuperiamo l'encoder relativo alla feature
        # trasformo il codice in numero e controllo che il codice sia valido

        for feature_name in self.features_input:
            code = features_dict[feature_name]
            encoder = self.dizionario_encoders[feature_name]

            if code not in encoder.classes_:
                encoded_value = 0 # Valore di default
            else:
                encoded_value = encoder.transform([code])[0]
            
            encoded_input.append(encoded_value)

        # passiamo la lista di input numerici al metodo 'predict' del nostro modello che restituisce un array con la predizione
        # essendo che la colonna edible/poisonous è la prima controllo solo il primo valore
        input_df = pd.DataFrame([encoded_input], columns=self.features_input)
        prediction_num = self.model.predict(input_df)[0]

        # ritrasformiamo il risultato numerico (0 o 1) nella sua etichetta testuale originale ('e' o 'p')
        result_text = self.le_target.inverse_transform([prediction_num])[0]

        return result_text


    def visualize_tree(self, output_filename="decision_tree.png"):
        """
        Genera e salva una visualizzazione dell'albero decisionale.
        """
        if not self.model:
            print("Il modello non è stato ancora addestrato. Impossibile visualizzare l'albero.")
            return

        # Aumenta la dimensione della figura per una migliore leggibilità
        plt.figure(figsize=(40, 20))
        
        import re
        from matplotlib.colors import to_rgba
        from matplotlib.text import Annotation

        annotations = tree.plot_tree(
            self.model,
            feature_names=self.features_input,
            class_names=list(self.le_target.classes_),
            filled=True,
            rounded=True,
            fontsize=10
        )
        
        # Definiamo i colori personalizzati
        colors = {'e': "#5D8053", 'p': "#C63636"}
        class_names = self.le_target.classes_

        for artist in annotations:
            # Applica i colori solo ai nodi (che sono oggetti Annotation)
            if not isinstance(artist, Annotation):
                continue

            # Ottieni il box del nodo
            box = artist.get_bbox_patch()
            if not box:
                continue

            # Estrai il 'value' dal testo del nodo per calcolare la purezza
            text = artist.get_text()
            match = re.search(r'value = \[([\d\s,.]+)\]', text)
            if not match:
                continue
            
            try:
                class_counts_str = match.group(1).split(',')
                class_counts = np.array([float(c.strip()) for c in class_counts_str])
            except (ValueError, IndexError):
                continue

            if np.sum(class_counts) == 0:
                continue

            # Determina la classe maggioritaria e il colore
            majority_class_idx = np.argmax(class_counts)
            majority_class_name = class_names[majority_class_idx]
            color_hex = colors[majority_class_name]
            
            # Calcola la purezza e l'alpha per la trasparenza
            purity = np.max(class_counts) / np.sum(class_counts)
            alpha = 0.6 + (purity - 0.5) * 0.8 if purity > 0.5 else 0.6
            
            # Applica il colore
            rgba_color = to_rgba(color_hex, alpha=alpha)
            box.set_facecolor(rgba_color)
        
        plt.title("Albero Decisionale per la Classificazione dei Funghi", fontsize=20)
        
        # Salva l'immagine nella cartella 'data'
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, output_filename)
        
        try:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"\nAlbero decisionale salvato in: {output_path}")
        except Exception as e:
            print(f"\nErrore durante il salvataggio dell'albero: {e}")
        finally:
            # Chiude la figura per liberare memoria e non mostrarla interattivamente
            plt.close()


if __name__ == '__main__':
    print("Esecuzione di poison_model.py come script principale...")
    try:
        # Assumiamo che 'mushrooms.csv' sia nella stessa cartella dello script
        print("1. Addestramento del classificatore di funghi...")
        classifier = MushroomClassifier(csv_path='mushrooms.csv')
        print("   Modello addestrato con successo.")
        
        print("\n2. Generazione della visualizzazione dell'albero decisionale...")
        classifier.visualize_tree(output_filename="decision_tree.png")
        print("   Visualizzazione completata.")

    except FileNotFoundError:
        print("\nERRORE: File 'mushrooms.csv' non trovato.")
        print("Per favore, esegui prima lo script 'data_setup.py' per scaricare il dataset.")
    except Exception as e:
        print(f"\nSi è verificato un errore inaspettato: {e}")
