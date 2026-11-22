import argparse
from clasificacion import *
from preprocesamiento import leer_imagen


def main():
    ap = argparse.ArgumentParser(description="Clasificador HOG + StandardScaler + KNN")
    ap.add_argument("imagen", nargs="?", help="Ruta de la imagen a clasificar")
    ap.add_argument("--dataset", default="PINK_IN_YOUR_AREA")
    ap.add_argument("--modelo", default=None, help="Archivo del modelo")
    ap.add_argument("--entrenar", action="store_true", help="Forzar reentrenamiento")
    args, unknown = ap.parse_known_args([]) # Forzar a ignorar argumentos de la línea de comandos de Jupyter

    ruta_modelo = Path(args.modelo or f"modelo_hog_{Path(args.dataset).name}.joblib")

    if args.entrenar or not ruta_modelo.exists():
        if not Path(args.dataset).exists():
            print(f"Para entrenar, se necesita la carpeta del dataset en '{args.dataset}'.")
        else:
            entrenar(args.dataset, ruta_modelo)

    if args.imagen:
        img = leer_imagen(args.imagen)
        if img is None:
            raise SystemExit(f"No se pudo leer la imagen: {args.imagen}")
        if not ruta_modelo.exists():
            raise SystemExit(f"No se encontró el modelo: {ruta_modelo}")

        clase, confianza, detalle = clasificar_imagen(img, ruta_modelo)
        print("\n=== RESULTADO DE CLASIFICACIÓN ===")
        print(f"Imagen: {args.imagen}")
        print(f"Predicción: {clase} (confianza: {confianza:.0%})")
        print("Detalle:", {c: f"{p:.0%}" for c, p in detalle.items()})


if __name__ == "__main__":
    main()

ruta_imagen = input("Ingresa la ruta de la imagen a clasificar (ej. /content/test.png): ")

if Path(ruta_imagen).exists():
    img_test = leer_imagen(ruta_imagen)
    if img_test is not None:
        # Asumimos el nombre del modelo por defecto según tu código anterior
        modelo_actual = "modelo_hog_PINK_IN_YOUR_AREA.joblib"
        try:
            clase_pred, conf_pred, similitudes = clasificar_imagen(img_test, modelo_actual)

            print("\n=== RESULTADO DE CLASIFICACIÓN ===")
            print(f"Imagen analizada: {ruta_imagen}")
            print(f"Predicción final: {clase_pred} (Confianza: {conf_pred:.2%})")
            print("\nPorcentajes de similitud:")
            for c, p in similitudes.items():
                print(f"  - {c}: {p:.2%}")
        except FileNotFoundError:
            print(f"Error: No se encontró el modelo '{modelo_actual}'. Por favor, entrena el modelo primero.")
        except Exception as e:
            print(f"Ocurrió un error al clasificar: {e}")
    else:
        print("Error: No se pudo leer la imagen. Verifica que sea un formato válido.")
else:
    print("Error: La ruta proporcionada no existe.")