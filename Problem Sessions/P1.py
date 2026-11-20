import numpy as np

# Dati del dataset
mesi = np.array([24, 30, 36])
pesi = np.array([13, 14, 16])

# Punto 4: Equazioni normali con il metodo dei minimi quadrati

# Creiamo la matrice X aggiungendo una colonna di 1 per il termine noto (intercetta b)
X = np.vstack([mesi, np.ones(len(mesi))]).T
Y = pesi

# Risolviamo il sistema lineare X^T * X * theta = X^T * Y
# dove theta contiene [a, b]
theta = np.linalg.solve(X.T @ X, X.T @ Y)
a, b = theta

print("--- Punto 4: Soluzione Minimi Quadrati ---")
print(f"Parametro a (pendenza): {a:.4f}")
print(f"Parametro b (intercetta): {b:.4f}")

# Calcolo dell'errore associato (SSE)
previsioni = X @ theta
errore_sse = np.sum((Y - previsioni) ** 2)
print(f"Errore quadratico totale (SSE): {errore_sse:.4f}")

# Punto 5: Stima a 25 e 34 mesi
eta_stime = np.array([25, 34])
# Applichiamo il modello y = a*x + b
pesi_stimati = a * eta_stime + b

print("\n--- Punto 5: Stime del peso ---")
print(f"Peso stimato a 25 mesi: {pesi_stimati[0]:.3f} kg")
print(f"Peso stimato a 34 mesi: {pesi_stimati[1]:.3f} kg")