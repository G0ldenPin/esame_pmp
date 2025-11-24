import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import LabelEncoder
# -----------------------------
# 1. Scaricare il dataset
# -----------------------------
mushroom = fetch_ucirepo(id=73) #scarica il dataset dei funghi (Mushroom) dal repository UCI.
X = mushroom.data.features #prende le features
y = mushroom.data.targets  #prende i target (commestibile o meno)
df = pd.concat([X, y], axis=1) #unisce in un singolo dataframe 
target_col = df.columns[-1] #memorizza colonna target(ultima colonna del df)

# -----------------------------
# 2️. Funzioni per i grafici
# -----------------------------
def grafico_target():
    plt.figure(figsize=(6,4)) #dim grafico
    sns.countplot(x=target_col, data=df) #conta occorrenze x classe
    plt.title("Distribuzione dei funghi commestibili e velenosi")
    plt.savefig("grafico_target.png")
    plt.show() #visualizza grafico
def grafico_feature(feature):
    if feature not in df.columns:  # controlla se la feature esiste nel DataFrame
        messagebox.showerror("Errore", "Feature non valida!")
        return  
    plt.figure(figsize=(8,5))  # crea una nuova figura con dimensioni 8x5 
    sns.countplot(x=feature, hue=target_col, data=df)# crea un grafico a barre (countplot) con Seaborn:
    plt.title(f"Distribuzione della feature '{feature}' per classe")
    plt.xticks(rotation=45)  # Ruota le etichette dell'asse X di 45 gradi
    plt.tight_layout()  # Adatta automaticamente la disposizione degli elementi del grafico
    plt.savefig(f"grafico_feature_{feature}.png")
    plt.show()  #mostra il grafico

def heatmap_correlazioni():
    # Copia del dataframe e codifica numerica delle feature 
    df_encoded = df.copy()
    # Creazione di un oggetto LabelEncoder per convertire dati categorici in numerici
    le = LabelEncoder()
    # Ciclo su tutte le colonne del dataframe per applicare la codifica
    for col in df_encoded.columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    # Calcolo correlazioni
    corr = df_encoded.corr()
    
    plt.figure(figsize=(14,12))
    
    # Heatmap con libreria Seaborn
    sns.heatmap(
        corr, 
        annot=True,               # mostra valori numerici
        fmt=".2f",                # formato numerico con 2 decimali
        cmap="viridis",           # palette di colori
        linewidths=0.5,           # bordi tra le celle
        linecolor="gray",         # colore dei bordi
        cbar_kws={"shrink": 0.8},# dimensione barra colori
        annot_kws={"size":10}     # dimensione numeri annotazioni
    )
    
    plt.title("🍄 Heatmap delle correlazioni tra le variabili dei funghi 🍄", fontsize=16, fontweight="bold")
    #etichette e dimensioni 
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("heatmap_correlazioni.png")
    plt.show()

# -----------------------------
# 3️. Funzioni GUI
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
# 4️. Creazione GUI principale
# -----------------------------
root = tk.Tk()
root.title("Visualizzatore Funghi 🍄")
root.geometry("400x300")

tk.Label(root, text="🍄 VISUALIZZATORE DI FUNGI 🍄", font=("Helvetica", 16, "bold")).pack(pady=20)

tk.Button(root, text="Distribuzione Target", width=25, command=grafico_target).pack(pady=10)
tk.Button(root, text="Distribuzione Feature", width=25, command=open_feature_window).pack(pady=10)
tk.Button(root, text="Heatmap Correlazioni", width=25, command=heatmap_correlazioni).pack(pady=10)
tk.Button(root, text="Esci", width=25, command=root.destroy).pack(pady=10)

root.mainloop()
