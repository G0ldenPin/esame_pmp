import pandas as pd
import io
import zipfile
import os
import argparse
import shutil
from ucimlrepo import fetch_ucirepo

# --- Costanti ---
# Lo script è progettato per risiedere nella cartella 'data'.

# Il percorso assoluto della directory 'data' è la directory in cui si trova questo script.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Calcola i percorsi delle altre cartelle di progetto
PROJECT_ROOT = os.path.abspath(os.path.join(DATA_DIR, '..'))
POISON_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, 'poison_analysis')
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, 'visualization')

# I percorsi per i file CSV e ZIP sono calcolati all'interno di DATA_DIR.
CSV_FILE = "mushrooms.csv"
ZIP_FILE = "mushroom.zip"
CSV_PATH = os.path.join(DATA_DIR, CSV_FILE)
ZIP_PATH = os.path.join(DATA_DIR, ZIP_FILE)

# Nomi delle colonne per il fallback da file .data (senza header)
COLUMN_NAMES = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]

# --- Funzioni ---

def download_from_url():
    """Scarica il dataset Mushroom utilizzando la libreria ucimlrepo."""
    print("Tentativo di download del dataset Mushroom tramite ucimlrepo...")
    try:
        # ID 73 corrisponde al dataset "Mushroom"
        mushroom_dataset = fetch_ucirepo(id=73)

        # I dati sono separati in features (X) and targets (y)
        X = mushroom_dataset.data.features
        y = mushroom_dataset.data.targets

        # Combiniamo features e target in un unico DataFrame
        df = pd.concat([y, X], axis=1)

        # Salviamo il DataFrame in un file CSV
        df.to_csv(CSV_PATH, index=False)

        print("\nSUCCESSO!")
        print(f"File '{CSV_PATH}' creato correttamente con {len(df)} righe.")
        return True

    except Exception as e:
        print("\nERRORE durante il download con ucimlrepo:")
        print(e)
        print("\nSe il problema persiste, prova l'alternativa manuale:")
        print(f"1. Scarica il dataset in formato ZIP (es. da Kaggle).")
        print(f"2. Salvalo come '{ZIP_PATH}'.")
        print(f"3. Esegui: python setup_dataset.py --zip")
        return False


def extract_from_zip():
    """Estrae il dataset da un file ZIP, cercando un file .csv o .data."""
    print(f"Tentativo di estrazione del dataset da '{ZIP_PATH}'...")
    if not os.path.exists(ZIP_PATH):
        print(f"\nERRORE: File '{ZIP_PATH}' non trovato.")
        print("Assicurati di aver scaricato il file ZIP e di averlo salvato nella cartella 'data'.")
        return False

    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            file_to_extract = None
            is_csv = False
            
            # Cerca prima un file .csv
            for name in zip_ref.namelist():
                if name.endswith('.csv'):
                    file_to_extract = name
                    is_csv = True
                    break
            
            # Se non trova .csv, cerca .data
            if not file_to_extract:
                for name in zip_ref.namelist():
                    if name.endswith('.data'):
                        file_to_extract = name
                        break

            if not file_to_extract:
                print(f"\nERRORE: Impossibile trovare un file '.csv' o '.data' in '{ZIP_PATH}'.")
                return False

            with zip_ref.open(file_to_extract) as f:
                if is_csv:
                    # Il CSV da Kaggle dovrebbe avere gli header
                    print(f"Trovato file '{file_to_extract}', lo leggo come CSV con header.")
                    df = pd.read_csv(f)
                else:
                    # Il file .data non ha header
                    print(f"Trovato file '{file_to_extract}', lo leggo come file .data senza header.")
                    content = f.read().decode('utf-8')
                    file_stream = io.StringIO(content)
                    df = pd.read_csv(file_stream, header=None, names=COLUMN_NAMES)

        # Salva il DataFrame nel percorso CSV finale
        df.to_csv(CSV_PATH, index=False)

        print("\nSUCCESSO!")
        print(f"File '{CSV_PATH}' creato correttamente da '{file_to_extract}' con {len(df)} righe.")
        return True

    except Exception as e:
        print(f"\nERRORE durante l'estrazione dallo ZIP:")
        print(e)
        return False

def copy_csv_to_projects():
    """Copia il file CSV nelle cartelle dei progetti che lo utilizzano."""
    if not os.path.exists(CSV_PATH):
        print(f"\nERRORE: Il file sorgente '{CSV_PATH}' non esiste. Impossibile copiare.")
        return

    destinations = {
        "Analisi dei veleni": POISON_ANALYSIS_DIR,
        "Visualizzazione": VISUALIZATION_DIR
    }

    print("\n--- Copia del dataset nelle altre cartelle ---")
    for name, dest_dir in destinations.items():
        if not os.path.exists(dest_dir):
            print(f"ATTENZIONE: La cartella del progetto '{name}' ('{dest_dir}') non esiste. Salto la copia.")
            continue
        
        try:
            shutil.copy(CSV_PATH, dest_dir)
            print(f"SUCCESSO: Copiato '{CSV_FILE}' in '{dest_dir}'.")
        except Exception as e:
            print(f"ERRORE: Impossibile copiare in '{dest_dir}'. Dettagli: {e}")

# --- Esecuzione ---

def main():
    # --- DEBUG INFO ---
    print(f"Directory di lavoro corrente: {os.getcwd()}")
    print(f"Percorso di salvataggio calcolato: {CSV_PATH}")
    # ------------------

    parser = argparse.ArgumentParser(description="Setup del dataset dei funghi.")
    parser.add_argument(
        '--zip',
        action='store_true',
        help=f"Se specificato, estrae il dataset da '{ZIP_FILE}' invece di scaricarlo."
    )
    args = parser.parse_args()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    success = False
    if args.zip:
        success = extract_from_zip()
    else:
        success = download_from_url()

    # Se il download o l'estrazione hanno avuto successo, copia il file
    if success:
        copy_csv_to_projects()

if __name__ == "__main__":
    main()
