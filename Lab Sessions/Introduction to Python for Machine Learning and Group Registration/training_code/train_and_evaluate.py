import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 1. Caricamento del dataset
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42)

# 2. Testare diversi valori di k
k_values = list(range(1, 31))
accuracies = []
f1_scores = []  # Scelta della metrica aggiuntiva: F1-score (macro)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='macro')  # average='macro' per multiclasse

    accuracies.append(acc)
    f1_scores.append(f1)

# 3. Salvataggio della figura
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, label='Accuracy', marker='o')
plt.plot(k_values, f1_scores, label='F1 Score (macro)', marker='s')
plt.title('kNN Performance vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.xticks(k_values)
plt.legend()
plt.grid(True)

# Assicurati che la cartella figures esista
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/knn_evaluation.png')
print("Figura salvata in: figures/knn_evaluation.png")

# 4. Selezione del k ottimale (basato sulla massima accuracy)
best_k_index = np.argmax(accuracies)
best_k = k_values[best_k_index]
best_accuracy = accuracies[best_k_index]

print(f"Miglior numero di vicini (k): {best_k} con Accuracy: {best_accuracy:.4f}")

# 5. Training del modello finale con il miglior k e salvataggio in .pkl
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)

# Assicurati che la cartella best_model esista
os.makedirs('../best_model', exist_ok=True)
joblib.dump(best_model, '../best_model/model.pkl')
print("Modello salvato in: best_model/model.pkl")