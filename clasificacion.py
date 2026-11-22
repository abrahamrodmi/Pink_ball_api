from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from preprocesamiento import *
import joblib
from pathlib import Path
import numpy as np

VALORES_VECINOS = [1, 3, 5, 7, 11]

def entrenar(raiz_dataset, ruta_modelo):
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = cargar_datos_y_dividir(raiz_dataset)

    print(f"Imagenes -> train: {len(y_tr)} | validation: {len(y_va)} | test: {len(y_te)}")
    print("Clases (train):", {c: int(n) for c, n in zip(*np.unique(y_tr, return_counts=True))})

    # StandardScaler
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_te)

    mejor = (-1, None, None)
    for n in VALORES_VECINOS:
        knn = KNeighborsClassifier(n_neighbors=n).fit(X_tr_scaled, y_tr)
        acc = accuracy_score(y_va, knn.predict(X_va_scaled))
        if acc > mejor[0]:
            mejor = (acc, n, knn)

    acc_va, n, knn = mejor
    print(f"\nMejor configuración: vecinos={n} (validation={acc_va:.3f})")

    pred = knn.predict(X_te_scaled)
    print(f"\n=== TEST ===\nExactitud: {accuracy_score(y_te, pred):.3f}")
    print(classification_report(y_te, pred))
    print("Matriz de confusión:")
    print(knn.classes_)
    print(confusion_matrix(y_te, pred, labels=knn.classes_))

    joblib.dump({"scaler": scaler, "knn": knn}, ruta_modelo)
    print(f"\nModelo guardado en {Path(ruta_modelo).resolve()}")


def clasificar_imagen(img, ruta_modelo):
    """Clasifica una imagen. Devuelve (clase, confianza, detalle)."""
    modelo = joblib.load(ruta_modelo)
    if img is None:
        raise ValueError("La imagen es None.")

    x = extraer_hog(img)
    x_scaled = modelo["scaler"].transform([x])

    knn = modelo["knn"]
    probas = knn.predict_proba(x_scaled)[0]
    i = probas.argmax()
    return str(knn.classes_[i]), float(probas[i]), {str(c): float(p) for c, p in zip(knn.classes_, probas)}