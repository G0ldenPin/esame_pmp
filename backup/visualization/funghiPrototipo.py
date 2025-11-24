import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#tutto mappato con nomi giusti, senza veil-type essendo sempre costante nel DF e con la rimozione dei valori "?"
# -----------------------------
# 1️⃣ Scaricare il dataset
# -----------------------------
mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
y = mushroom.data.targets
df = pd.concat([X, y], axis=1)
target_col = df.columns[-1]

# -----------------------------
# 1.1️⃣ Rimuovere i valori mancanti "?" nella colonna stalk-root
# -----------------------------
df = df[df['stalk-root'] != '?']

# -----------------------------
# 1.2️⃣ Mapping delle lettere ai nomi descrittivi
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
    "stalk-root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs"},
    "stalk-surface-above-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-surface-below-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-color-above-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "stalk-color-below-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "veil-type": {"p": "partial"},
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

if 'veil-type' in df.columns:
    df = df.drop(columns=['veil-type'])
# -----------------------------
# 2️⃣ Funzioni per i grafici
# -----------------------------
def grafico_target():
    plt.figure(figsize=(6,4))
    palette_color = ["#4CAF50", "#FF9800"] 
    sns.countplot(x=target_col, data=df, palette=palette_color)
    plt.title("Distribuzione dei funghi commestibili e velenosi")
    plt.savefig("prototipo_grafico_target.png")
    plt.show()

def grafico_feature(feature):
    if feature not in df.columns:
        messagebox.showerror("Errore", "Feature non valida!")
        return
    plt.figure(figsize=(8,5))
    palette_color = ["#4CAF50", "#FF9800"] 
    sns.countplot(x=feature, hue=target_col, data=df ,palette=palette_color)
    plt.title(f"Distribuzione della feature '{feature}' per classe")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"prototipo_grafico_feature_{feature}.png")
    plt.show()

def heatmap_correlazioni():
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    corr = df_encoded.corr()
    plt.figure(figsize=(14,12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis",
                linewidths=0.5, linecolor="gray", cbar_kws={"shrink":0.8}, annot_kws={"size":10})
    plt.title("Heatmap delle correlazioni tra le variabili dei funghi", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("prototipo_heatmap_correlazioni.png")
    plt.show()

# -----------------------------
# 3️⃣ Funzione KMeans++
# -----------------------------
def kmeans_cluster(n_clusters=2):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(df_encoded.drop(columns=[target_col]))
    df_encoded['Cluster'] = clusters

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(df_encoded.drop(columns=[target_col, 'Cluster']))

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=pca_result[:,0],
        y=pca_result[:,1],
        hue=df_encoded['Cluster'],
        palette="Set2",
        style=df[target_col],
        s=100
    )
    plt.title(f"KMeans++ Clustering con {n_clusters} cluster")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster / Target", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"prototipo_kmeans_{n_clusters}_clusters.png")
    plt.show()

# -----------------------------
# 4️⃣ GUI Feature
# -----------------------------
def open_feature_window():
    feature_win = tk.Toplevel(root)
    feature_win.title("Seleziona Feature")
    tk.Label(feature_win, text="Scegli una feature da visualizzare:").pack(pady=5)
    features = df.columns.drop(target_col)
    feature_var = tk.StringVar()
    combo = ttk.Combobox(feature_win, values=list(features), textvariable=feature_var, state="readonly")
    combo.pack(padx=10, pady=10)

    def show_feature_plot():
        selected = feature_var.get()
        if selected:
            grafico_feature(selected)
        else:
            messagebox.showwarning("Attenzione", "Seleziona una feature!")

    tk.Button(feature_win, text="Mostra grafico", command=show_feature_plot).pack(pady=10)

# -----------------------------
# 5️⃣ Creazione GUI principale migliorata
# -----------------------------
root = tk.Tk()
root.title("🍄Visualizzatore Funghi🍄")
root.geometry("397x400")
root.configure(bg="#f2f2f2")

cluster_var = tk.IntVar(value=4)  # variabile cluster

# Titolo principale
title_frame = tk.Frame(root, bg="#E05151", pady=15, padx=10)
title_frame.pack(fill="x")
tk.Label(
    title_frame,
    text="🍄 VISUALIZZATORE DI FUNGI 🍄",
    font=("Helvetica", 18, "bold"),
    fg="white",
    bg="#E05151"
).pack()

# Frame bottoni
button_frame = tk.Frame(root, bg="#f2f2f2")
button_frame.pack(pady=15)

btn_style = {
    "width": 30,
    "bg": "white",
    "fg": "#333333",
    "activebackground": "#E05151",
    "activeforeground": "white",
    "font": ("Helvetica", 10, "bold"),
    "bd": 1,
    "relief": "raised"
}

tk.Button(button_frame, text="Distribuzione Target", command=grafico_target, **btn_style).pack(pady=5)
tk.Button(button_frame, text="Distribuzione Feature", command=open_feature_window, **btn_style).pack(pady=5)
tk.Button(button_frame, text="Heatmap Correlazioni", command=heatmap_correlazioni, **btn_style).pack(pady=5)

# Cluster
cluster_frame = tk.Frame(root, bg="#f2f2f2")
cluster_frame.pack(pady=10)
tk.Button(cluster_frame, text="Esegui KMeans++", command=lambda: kmeans_cluster(cluster_var.get()), **btn_style).pack(pady=5)
tk.Label(cluster_frame, text="Numero di cluster KMeans++:", bg="#f2f2f2", font=("Helvetica", 10)).pack()
tk.Entry(cluster_frame, textvariable=cluster_var, width=10).pack(pady=5)

# Bottone Esci
tk.Button(root, text="Esci", command=root.destroy, **btn_style).pack(pady=15)

root.mainloop()
