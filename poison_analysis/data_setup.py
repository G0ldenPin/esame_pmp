import pandas as pd
import io
import requests

print("Tentativo di download del dataset Mushroom dalla UCI Repository...")

# 1. URL diretto ai dati grezzi (Raw Data)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

# 2. Definiamo i nomi delle colonne manualmente
# (Il file originale non li ha, quindi glieli diamo noi per far funzionare il tuo script)
column_names = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat"
]

try:
    # 3. Scarichiamo e leggiamo i dati
    response = requests.get(url)
    response.raise_for_status()  # Controlla se il download è andato bene

    # 4. Creiamo il DataFrame
    file_stream = io.StringIO(response.content.decode('utf-8'))
    df = pd.read_csv(file_stream, header=None, names=column_names)

    # 5. Salviamo il file CSV pulito nella cartella corrente
    df.to_csv("mushrooms.csv", index=False)

    print("\n✅ SUCCESSO!")
    print(f"File 'mushrooms.csv' creato correttamente con {len(df)} righe.")
    print("Ora puoi eseguire il tuo script principale 'poison_tester.py'.")

except Exception as e:
    print("\n❌ ERRORE durante il download:")
    print(e)
    print("\nAlternativa manuale:")
    print("1. Vai su https://www.kaggle.com/uciml/mushroom-classification")
    print("2. Scarica il file, rinominalo 'mushrooms.csv' e mettilo qui.")