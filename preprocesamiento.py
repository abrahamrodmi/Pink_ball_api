from pathlib import Path
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

EXTENSIONES = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def leer_imagen(ruta):
    """Lee la imagen. Si tiene transparencia, la compone sobre fondo blanco."""
    img = cv.imread(str(ruta), cv.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alfa = img[:, :, 3:4].astype(np.float32) / 255.0
        fondo = np.full_like(img[:, :, :3], 255, dtype=np.uint8)
        img = (img[:, :, :3] * alfa + fondo * (1 - alfa)).astype(np.uint8)
    return img


def extraer_hog(img):
    """Convierte a escala de grises, redimensiona y extrae características HOG."""
    # Escala de grises
    gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Redimensionar a un tamaño estándar para que HOG devuelva vectores del mismo tamaño
    gris = cv.resize(gris, (64, 128))
    hog = cv.HOGDescriptor()
    caracteristicas = hog.compute(gris)
    return caracteristicas.flatten()


def cargar_datos_y_dividir(raiz_dataset):
    """
    Lee directamente de las subcarpetas (ej. Jigglypuff, kirby, Wigglytuff)
    y divide en train (80%), val (10%) y test (10%).
    """
    raiz = Path(raiz_dataset)
    descs, etiquetas = [], []

    if not raiz.exists():
        raise FileNotFoundError(f"No se encontró la carpeta: {raiz.resolve()}")

    for carpeta_clase in sorted(p for p in raiz.iterdir() if p.is_dir()):
        archivos = sorted({r for ext in EXTENSIONES for r in carpeta_clase.glob(ext)})
        for ruta in archivos:
            img = leer_imagen(ruta)
            if img is None:
                continue
            descs.append(extraer_hog(img))
            etiquetas.append(carpeta_clase.name)

    X = np.array(descs)
    y = np.array(etiquetas)

    if len(X) == 0:
        raise ValueError(f"No se encontraron imágenes en {raiz.resolve()}")

    # 80% train, 20% temporal (para val y test)
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    # Dividir el 20% temporal en dos mitades iguales -> 10% val, 10% test
    X_va, X_te, y_va, y_te = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)