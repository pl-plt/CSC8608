import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Chargement des données
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # 0: Maligne, 1: Bénigne

# 2. Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalisation (Crucial pour l'interprétabilité des coefficients)
scaler = StandardScaler()
# Adapter le scaler sur les données d'entraînement et les transformer
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Entraînement du modèle "Glass-box"
# On utilise une pénalité L2 standard
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Évaluation rapide
y_pred = model.predict(X_test_scaled)
print(f"Accuracy de la Régression Logistique : {accuracy_score(y_test, y_pred):.4f}")

# 5. Extraction de l'explication (Intrinsèque)
coefficients = model.coef_[0]

# Création d'un DataFrame pour faciliter la manipulation
feature_importance = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': coefficients
})

# Tri par importance absolue
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=True)

# 6. Visualisation
plt.figure(figsize=(10, 8))
# On colorie en rouge les coefficients négatifs (poussent vers la classe 0: Maligne)
# et en bleu les positifs (poussent vers la classe 1: Bénigne)
colors = ['red' if c < 0 else 'blue' for c in feature_importance['Coefficient']]

plt.barh(feature_importance['Feature'][-15:], feature_importance['Coefficient'][-15:], color=colors[-15:])
plt.xlabel('Valeur du Coefficient (\u03b2)')
plt.title('Top 15 - Importance des variables (Régression Logistique)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()

output_filename = "glassbox_coefficients.png"
plt.savefig(output_filename)
print(f"Graphique sauvegardé dans {output_filename}")
