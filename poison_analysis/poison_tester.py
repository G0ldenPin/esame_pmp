import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import sys

# ==========================================
# 1. CONFIGURAZIONE E ADDESTRAMENTO
# ==========================================
print("Caricamento del cervello elettronico in corso...")

# Carica il dataset
try:
    df = pd.read_csv('mushrooms.csv')
except FileNotFoundError:
    print("ERRORE: Non trovo il file 'mushrooms.csv'. Assicurati che sia nella stessa cartella.")
    sys.exit()

# Le feature esatte che useremo
features_input = [
    'cap-shape',  # Forma cappello
    'cap-color',  # Colore cappello
    'gill-color',  # Colore lamelle
    'stalk-shape',  # Forma gambo
    'veil-type',  # Tipo di velo
    'veil-color',  # Colore del velo
    'odor'  # Odore
]

# --- CORREZIONE 1: Aggiunto .copy() per eliminare i warning rossi ---
X = df[features_input].copy()
y = df['class']

# Addestramento degli Encoder
dizionario_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    dizionario_encoders[col] = le

# Addestramento Encoder del Target
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Creiamo l'Albero Decisionale
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

print("Sistema pronto! Rispondi alle domande guardando il fungo.")

# ==========================================
# 2. DIZIONARIO DI TRADUZIONE CORRETTO
# ==========================================
# --- CORREZIONE 2: Sintassi del dizionario sistemata ---
menu_opzioni = {
    "cap-shape": {
        "A campana": "b", "Conico": "c", "Convesso": "x", "Piatto": "f",
        "Nodoso/Umbone": "k", "Infossato": "s"
    },
    "cap-color": {
        "Marrone": "n", "Giallo pallido (Buff)": "b", "Cannella": "c", "Grigio": "g",
        "Verde": "r", "Rosa": "p", "Viola": "u", "Rosso": "e", "Bianco": "w", "Giallo": "y"
    },
    "gill-color": {
        "Nero": "k", "Marrone": "n", "Giallo pallido": "b", "Cioccolato": "h", "Grigio": "g",
        "Verde": "r", "Arancione": "o", "Rosa": "p", "Viola": "u", "Rosso": "e", "Bianco": "w", "Giallo": "y"
    },
    "stalk-shape": {
        "Si allarga alla base": "e", "Si restringe (affusolato)": "t"
    },
    "veil-type": {
        "Parziale": "p", "Universale": "u"
    },
    "veil-color": {
        "Marrone": "n", "Arancione": "o", "Bianco": "w", "Giallo": "y"
    },
    "odor": {
        "Mandorla": "a", "Anice": "l", "Chimico/Creosoto": "c", "Pesce": "y",
        "Fetido": "f", "Muffa": "m", "Nessun odore": "n", "Pungente": "p", "Speziato": "s"
    }
}

# ==========================================
# 3. INTERAZIONE CON L'UTENTE (IL PEZZO MANCANTE)
# ==========================================

input_utente_numerico = []

print("\n" + "-" * 40)
print(" IDENTIFICAZIONE FUNGO (Visuale + Olfatto)")
print("-" * 40)

# Ciclo attraverso ogni caratteristica per fare le domande
for feature in features_input:
    print(f"\nDOMANDA: Com'è {feature.upper()}?")

    opzioni_disponibili = menu_opzioni[feature]
    chiavi_menu = list(opzioni_disponibili.keys())

    # Mostra menu
    for i, nome_opzione in enumerate(chiavi_menu):
        print(f"{i + 1}. {nome_opzione}")

    # Input utente
    while True:
        try:
            risposta = input("Inserisci il numero corrispondente: ")
            scelta = int(risposta) - 1

            if 0 <= scelta < len(chiavi_menu):
                risposta_italiano = chiavi_menu[scelta]
                codice_dataset = opzioni_disponibili[risposta_italiano]

                encoder_corrente = dizionario_encoders[feature]

                # Fallback di sicurezza se l'opzione non esiste nel training
                if codice_dataset not in encoder_corrente.classes_:
                    valore_codificato = 0
                else:
                    valore_codificato = encoder_corrente.transform([codice_dataset])[0]

                input_utente_numerico.append(valore_codificato)
                break
            else:
                print("Numero non valido. Riprova.")
        except ValueError:
            print("Per favore inserisci un numero valido.")

# ==========================================
# 4. PREDIZIONE E RISULTATO
# ==========================================

try:
    predizione_num = model.predict([input_utente_numerico])[0]
    risultato_testo = le_target.inverse_transform([predizione_num])[0]

    print("\n\n")
    print("=" * 40)
    if risultato_testo == 'e':
        print("✅ RISULTATO ANALISI: COMMESTIBILE")
    else:
        print("☠️  RISULTATO ANALISI: VELENOSO")
    print("=" * 40)

except Exception as e:
    print(f"Errore durante la predizione: {e}")

# Impedisce alla finestra di chiudersi subito su Windows/alcuni IDE
input("\nPremi INVIO per chiudere il programma...")