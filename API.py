from fastapi import FastAPI, UploadFile, File
import uvicorn
import joblib
import numpy as np
import cv2 as cv
from clasificacion import extraer_hog
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Clasificador de Imágenes HOG+KNN")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  
    allow_methods=["*"],
    allow_headers=["*"],
)



RUTA_MODELO = "modelo_hog_PINK_IN_YOUR_AREA.joblib"
try:
    modelo_cargado = joblib.load(RUTA_MODELO)
    print("Modelo cargado correctamente en la API.")
except FileNotFoundError:
    print("Advertencia: No se encontró el modelo. Entrena primero el algoritmo.")
    modelo_cargado = None

@app.post("/predict/")
async def predecir_imagen(file: UploadFile = File(...)):
    if modelo_cargado is None:
        return {"error": "El modelo no está disponible en el servidor."}

    # 1. Leer los bytes de la imagen subida
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    if img is None:
        return {"error": "Formato de imagen inválido o corrupto."}

    # 2. Extraer características y escalar
    x = extraer_hog(img)
    x_scaled = modelo_cargado["scaler"].transform([x])

    # 3. Predicción con KNN
    knn = modelo_cargado["knn"]
    probas = knn.predict_proba(x_scaled)[0]
    i = probas.argmax()

    # 4. Retornar el resultado en formato JSON
    return {
        "prediccion": str(knn.classes_[i]),
        "confianza": float(probas[i]),
        "detalle": {str(c): float(p) for c, p in zip(knn.classes_, probas)}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)