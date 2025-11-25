
# -----------------------------
# import
# -----------------------------
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from kmodes.kmodes import KModes

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import chi2_contingency

# -----------------------------
# Scaricare il dataset
# -----------------------------
import pandas as pd

df = pd.read_csv("mushrooms.csv")
#target_col = "class"
df = df.rename(columns={"posionous": "poisonous"})
target_col = "poisonous"
# -----------------------------
# Mapping
# -----------------------------
mappings = {
    "cap-shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken"},
    "cap-surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap-color": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "bruises": {"t": "bruises", "f": "no bruises"},
    "odor": {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul", "m": "musty", "n": "none", "p": "pungent", "s": "spicy"},
    "gill-attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill-spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill-size": {"b": "broad", "n": "narrow"},
    "gill-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray", "r": "green", "o": "orange", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "stalk-shape": {"e": "enlarging", "t": "tapering"},
    "stalk-root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs","?":"missing","r": "rooted"},
    "stalk-surface-above-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-surface-below-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-color-above-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "stalk-color-below-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "veil-type": {"p": "partial","u":"universal"},
    "veil-color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
    "ring-number": {"n": "none", "o": "one", "t": "two"},
    "ring-type": {"c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large", "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"},
    "spore-print-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green", "o": "orange", "u": "purple", "w": "white", "y": "yellow"},
    "population": {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered", "v": "several", "y": "solitary"},
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "u": "urban", "w": "waste", "d": "woods"},
    target_col: {"e": "edible", "p": "poisonous"}
}
for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)
        
#Rimozione veil-type (uguale per tutti i funghi)
#Nella descrizione del dataset era inizialmente "universale" & "parziale"
if 'veil-type' in df.columns:
    df = df.drop(columns=['veil-type'])

# -----------------------------
# grafici
# -----------------------------
def grafico_target():
    #Definire la figura
    plt.figure(figsize=(6,4)) 
    palette_color = ["#C63636", "#5D8053"]
    sns.countplot(x=target_col, data=df, hue=target_col, palette=palette_color) #Creazione grafico a barre
    plt.legend([],[], frameon=False) #Rimuovere legenda duplicata
    plt.title("Distribuzione dei funghi commestibili e velenosi")
    plt.xlabel(target_col.capitalize()) #maiuscola        
    plt.ylabel("Conteggio")
    plt.show()          

def grafico_feature(feature):
    # Controllare se la feature selezionata esiste nel dataframe
    if feature not in df.columns:        
        messagebox.showerror("Errore", "Feature non valida!")
        return
    
    plt.figure(figsize=(8,5)) #Definire la figura          
    palette_color = ["#C63636", "#5D8053"]
    sns.countplot(x=feature, hue=target_col, data=df, palette=palette_color) #Creazione grafico a barre 

    #Impostare il titolo con la feature corretta,etichette ed ottimizzazione degli spazi       
    plt.title(f"Distribuzione della feature '{feature}' per classe")                
    plt.xticks(rotation=45)            
    plt.tight_layout()            
    plt.show() 

def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.sum().sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def heatmap_correlazioni():
    cols = df.columns
    n = len(cols)

    M = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            M.iloc[i, j] = cramers_v(df[c1], df[c2])


    colors = ["#C63636", "#FFD9A5", "#5D8053"]
    cmap = LinearSegmentedColormap.from_list("verde_arancione", colors)


    plt.figure(figsize=(14,12)) #Creazione una nuova figura  
    
    #Creazione della heatmap con Seaborn
    sns.heatmap(
        M,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9}
    )

    plt.title("Correlazioni tra variabili", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()        

# -----------------------------
# K-MODES
# -----------------------------
def k_modes_cluster(n_clusters=2):
    df_encoded = df.copy() #Creazione di una copia per evitare di sostituire valori
    le = LabelEncoder() #Creazione di un oggetto label encoder

    #Codifica delle colonne 
    for col in df_encoded.columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    # Inizializzare l'algoritmo K-Modes
    # init='Huang' = metodo di inizializzazione
    # n_init=5 = numero di inizializzazioni diverse per trovare la soluzione migliore
    km = KModes(n_clusters=n_clusters, init='Cao', n_init=5, verbose=0)   #verbose:Disattiva messaggi di log

    #Applica K-Modes sul dataset escludendo la colonna target
    clusters = km.fit_predict(df_encoded.drop(columns=[target_col]))

    #Aggiunge i cluster trovati come nuova colonna
    df_encoded["Cluster"] = clusters

    #Serve per visualizzare i dati in un grafico 2D
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(df_encoded.drop(columns=[target_col, "Cluster"]))

    plt.figure(figsize=(8,6))

    #Disegno dei punti
    # Rinominiamo la colonna target per avere una legenda più chiara
    df_plot = df.rename(columns={target_col: "Tipo di fungo"})

    sns.scatterplot(
        x=pca_result[:,0],
        y=pca_result[:,1],
        hue=df_encoded["Cluster"],
        palette=["#C63636", "#5D8053", "#E08B51", "#70A4A5"],
        style=df_plot["Tipo di fungo"], # Usiamo la colonna rinominata
        s=100,
        markers={"edible": "o", "poisonous": "X"}
    )

    plt.title(f"K-Modes Clustering con {n_clusters} cluster")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.show()

# -----------------------------
# GUI 
# -----------------------------
def open_feature_window():
    #Aprire finestra da visualizzare
    feature_win = tk.Toplevel(root) #Nuova finestra
    feature_win.title("Seleziona Feature")
    tk.Label(feature_win, text="Scegli una feature da visualizzare:").pack(pady=5)

    features = df.columns.drop(target_col)
    feature_var = tk.StringVar() #Variabile per salvare la scelta dell'utente
    combo = ttk.Combobox(feature_win, values=list(features), textvariable=feature_var, state="readonly")#Tendina features
    combo.pack(padx=10, pady=10)
    #Funzione che mostra il grafico della feature selezionata
    def show_feature_plot():
        selected = feature_var.get()
        if selected:
            grafico_feature(selected)
        else:
            messagebox.showwarning("Attenzione", "Seleziona una feature!")

    tk.Button(feature_win, text="Mostra grafico", command=show_feature_plot).pack(pady=10)

# -----------------------------
# GUI principale
# -----------------------------
# Crea la finestra principale di Tkinter
root = tk.Tk()
root.title("🍄Visualizzatore Funghi🍄")  # Imposta il titolo della finestra
root.geometry("397x560")                 # Imposta dimensioni della finestra
root.configure(bg="#f2f2f2")            # Imposta il colore di sfondo della finestra

# Variabile Tkinter per il numero di cluster da inserire
cluster_var = tk.IntVar(value=4)        # Valore iniziale = 4

# ---------------------- TITOLO ----------------------
# Crea un frame per il titolo con sfondo rosso e padding
title_frame = tk.Frame(root, bg="#E05151", pady=15, padx=10)
title_frame.pack(fill="x")               # Riempie tutta la larghezza della finestra

# Aggiunge un'etichetta (Label) al frame del titolo
tk.Label(
    title_frame,
    text="🍄 VISUALIZZATORE DI FUNGHI 🍄",
    font=("Helvetica", 18, "bold"),     # Font e dimensione
    fg="white",                          # Colore testo bianco
    bg="#E05151"                         # Sfondo rosso (uguale al frame)
).pack()

# ---------------------- FRAME PULSANTI ----------------------
button_frame = tk.Frame(root, bg="#f2f2f2")  # Frame per i pulsanti principali
button_frame.pack(pady=15)                    # Padding verticale

# Stile condiviso per tutti i pulsanti
btn_style = {
    "width": 30,                # Larghezza del pulsante
    "bg": "white",              # Sfondo bianco
    "fg": "#333333",            # Colore testo grigio scuro
    "activebackground": "#E05151",  # Sfondo quando cliccato
    "activeforeground": "white",    # Testo quando cliccato
    "font": ("Helvetica", 10, "bold"),
    "bd": 1,                    # Bordo
    "relief": "raised"          # Stile del bordo
}

# Pulsante target
tk.Button(button_frame, text="Distribuzione Target", command=grafico_target, **btn_style).pack(pady=5)

# Pulsante feature 
tk.Button(button_frame, text="Distribuzione Feature", command=open_feature_window, **btn_style).pack(pady=5)

# Pulsante per mostrare la heatmap delle correlazioni
tk.Button(button_frame, text="Heatmap Correlazioni", command=heatmap_correlazioni, **btn_style).pack(pady=5)

# ---------------------- FRAME K-MODES ----------------------
cluster_frame = tk.Frame(root, bg="#f2f2f2")  # Frame per i pulsanti K-Modes
cluster_frame.pack(pady=10)

# Pulsante  clustering K-Modes
tk.Button(
    cluster_frame,
    text="Esegui K-Modes",
    command=lambda: k_modes_cluster(cluster_var.get()),  # Passa il numero di cluster scelto
    **btn_style
).pack(pady=5)

# Label per inserire il numero di cluster
tk.Label(cluster_frame, text="Numero di cluster:", bg="#f2f2f2", font=("Helvetica", 10)).pack()
tk.Entry(cluster_frame, textvariable=cluster_var, width=10).pack(pady=5)

# Pulsante uscita
tk.Button(root, text="Esci", command=root.destroy, **btn_style).pack(pady=15)

# Avvia il loop principale di Tkinter (mostra la finestra e gestisce eventi)
root.mainloop()