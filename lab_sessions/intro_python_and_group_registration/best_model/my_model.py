import joblib
import pandas as pd
import os


def predict(data):
    """
    Applica la funzione di predizione al dataset usando il modello salvato.

    Parametri:
    data : array-like, DataFrame pandas o array numpy.
           I dati su cui effettuare le predizioni.

    Ritorna:
    predictions : array numpy con le predizioni del modello.
    """
    # Percorso del modello salvato
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

    # Carica il modello
    model = joblib.load(model_path)

    # Se data è un DataFrame pandas, convertilo in array numpy
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    # Effettua le predizioni
    predictions = model.predict(data)

    return predictions


# --- Esempio di utilizzo ---
if __name__ == "__main__":
    # Esempio: carichiamo il dataset digits e testiamo i primi 5 elementi
    from sklearn.datasets import load_digits

    digits = load_digits()

    # Predizione usando la nostra funzione
    preds = predict(digits.data[:5])

    print("Predizioni per i primi 5 esempi:", preds)
    print("Classi reali:", digits.target[:5])