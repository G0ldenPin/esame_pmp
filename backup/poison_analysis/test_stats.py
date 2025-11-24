import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# 1. PREPARAZIONE (Stessa di prima)
df = pd.read_csv('mushrooms.csv')

features_input = [
    'cap-shape', 'cap-color', 'gill-color',
    'stalk-shape', 'veil-type', 'veil-color', 'odor'
]

X = df[features_input].copy()
y = df['class']

for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

le_target = LabelEncoder()
y = le_target.fit_transform(y)

# 2. SPLIT E TRAINING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. CALCOLO STATISTICHE
report = classification_report(y_test, y_pred, target_names=['Commestibile', 'Velenoso'])
cm = confusion_matrix(y_test, y_pred)
scores = cross_val_score(model, X, y, cv=5)
accuracy_cv = scores.mean() * 100
std_cv = scores.std() * 100

# Testo completo da salvare
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

# 4. SALVATAGGIO SU FILE
nome_file = "report_statistiche.txt"
with open(nome_file, "w") as f:
    f.write(testo_output)

print(f"\n✅ Fatto! Ho salvato tutti i dati nel file: {nome_file}")
print("Apri quel file per vedere le percentuali.")

# 5. GRAFICO (Migliorato con Titolo Informativo)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Set2', annot_kws={"size": 14},
            xticklabels=['Pred: Commestibile', 'Pred: Velenoso'],
            yticklabels=['Reale: Commestibile', 'Reale: Velenoso'])

plt.xlabel('Predizione Modello', fontsize=12)
plt.ylabel('Realtà', fontsize=12)
# Mettiamo l'accuratezza direttamente nel titolo del grafico
plt.title(f'Matrice di Confusione\nAccuratezza Media: {accuracy_cv:.2f}%', fontsize=15)
plt.tight_layout()
plt.savefig("matrice_di_confusione.png")
plt.show()
