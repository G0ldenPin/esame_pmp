import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
from matplotlib.colors import LinearSegmentedColormap

# --- 1. PREPARAZIONE DEI DATI ---

# Definiamo il percorso del file CSV. Lo script si aspetta che 'mushrooms.csv'
# sia nella stessa cartella.
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'mushrooms.csv')
df = pd.read_csv(csv_path)

# Definiamo la cartella di output per i risultati.
# I risultati verranno salvati nella cartella 'data' del progetto.
output_dir = os.path.join(script_dir, '..', 'data')
os.makedirs(output_dir, exist_ok=True)

# Selezioniamo le stesse feature utilizzate per l'addestramento del modello.
features_input = [
    'cap-shape', 'cap-color', 'gill-color',
    'stalk-shape', 'veil-type', 'veil-color', 'odor'
]

# Separiamo le feature (X) dal target (y).
X = df[features_input].copy()
y = df['poisonous']

# Codifichiamo le feature categoriali in valori numerici.
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Codifichiamo anche il target.
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# --- 2. DIVISIONE DEL DATASET E ADDESTRAMENTO DEL MODELLO ---

# Dividiamo il dataset in un set di addestramento (70%) e un set di test (30%).
# 'random_state=42' assicura che la divisione sia sempre la stessa.
# 'stratify=y' mantiene la stessa proporzione di funghi velenosi e commestibili
# in entrambi i set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Creiamo e addestriamo il modello di albero decisionale.
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Eseguiamo le predizioni sul set di test.
y_pred = model.predict(X_test)

# --- 3. CALCOLO DELLE STATISTICHE DI VALUTAZIONE ---

# Generiamo un report di classificazione che include precisione, richiamo e F1-score.
report = classification_report(y_test, y_pred, target_names=['Commestibile', 'Velenoso'])

# Calcoliamo la matrice di confusione per vedere gli errori in dettaglio.
cm = confusion_matrix(y_test, y_pred)

# Eseguiamo una cross-validation a 5 fold per una stima più robusta dell'accuratezza.
scores = cross_val_score(model, X, y, cv=5)
accuracy_cv = scores.mean() * 100
std_cv = scores.std() * 100

# --- 4. CREAZIONE E SALVATAGGIO DEL REPORT TESTUALE ---

# Creiamo una stringa formattata con tutti i risultati.
testo_output = f"""
=============================================
REPORT STATISTICO COMPLETO - MUSHROOM DATASET
=============================================

1. ACCURATEZZA (Cross-Validation 5-fold)
----------------------------------------
Media: {accuracy_cv:.2f}%
Deviazione Standard: +/- {std_cv:.2f}%

2. REPORT DI CLASSIFICAZIONE
----------------------------------------
{report}

3. DETTAGLIO ERRORI (Matrice di Confusione)
----------------------------------------
Su un totale di {len(y_test)} funghi testati:
- Veri Negativi (Commestibili corretti): {cm[0][0]}
- Veri Positivi (Velenosi corretti):     {cm[1][1]}
- Falsi Positivi (Allarmi inutili):      {cm[0][1]}
- Falsi Negativi (PERICOLOSI):           {cm[1][0]} <--- Questo numero deve essere 0!
"""

# Salviamo il report in un file di testo.
report_path = os.path.join(output_dir, "report_statistiche.txt")
with open(report_path, "w") as f:
    f.write(testo_output)

print(f"\n✅ Fatto! Ho salvato tutti i dati nel file: {report_path}")
print("Apri quel file per vedere le percentuali.")

# --- 5. CREAZIONE E SALVATAGGIO DELLA MATRICE DI CONFUSIONE GRAFICA ---

# Creiamo un grafico della matrice di confusione per una visualizzazione più chiara.
plt.figure(figsize=(10, 7))

# Creiamo la colormap personalizzata
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_heatmap',
    ["#C63636", "#FFD9A5", "#5D8053"]
)

sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, annot_kws={"size": 14},
            xticklabels=['Pred: Commestibile', 'Pred: Velenoso'],
            yticklabels=['Reale: Commestibile', 'Reale: Velenoso'])

plt.xlabel('Predizione Modello', fontsize=12)
plt.ylabel('Realtà', fontsize=12)
plt.title(f'Matrice di Confusione\nAccuratezza Media: {accuracy_cv:.2f}%', fontsize=15)
plt.tight_layout()

# Salviamo il grafico come immagine PNG.
plot_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(plot_path)
plt.close()
