import numpy as np


def check_input_format(data):
    """
    Funzione ausiliaria per controllare che i dati in input abbiano il formato corretto.
    """
    if isinstance(data, list):
        data = np.array(data)

    if not isinstance(data, np.ndarray):
        raise ValueError("I dati di input devono essere array numpy, liste o DataFrame pandas.")

    if data.ndim != 2:
        raise ValueError("I dati di input devono essere bidimensionali (n_campioni, n_features).")

    print(f"Input validato correttamente: {data.shape[0]} campioni, {data.shape[1]} features.")
    return data