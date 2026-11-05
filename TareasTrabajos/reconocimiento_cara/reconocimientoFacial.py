"""Detecta rostros con Haar en la cámara web o en un archivo de video."""

import argparse
from pathlib import Path

import cv2 as cv


def main():
    parser = argparse.ArgumentParser(
        description="Detecta rostros y guarda cada recorte como una imagen JPG."
    )
    parser.add_argument(
        "origen",
        nargs="?",
        default="0",
        help="índice de cámara (0 por defecto) o ruta de un video, por ejemplo caras.mp4",
    )
    parser.add_argument(
        "-o", "--salida", default="rostros", help="carpeta para guardar los recortes"
    )
    args = parser.parse_args()

    origen = int(args.origen) if args.origen.isdecimal() else args.origen
    cap = cv.VideoCapture(origen)
    if not cap.isOpened():
        parser.error(f"No se pudo abrir la cámara o el video: {args.origen}")

    cascade_path = Path(cv.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
    detector = cv.CascadeClassifier(str(cascade_path))
    if detector.empty():
        cap.release()
        parser.error(f"No se pudo cargar el clasificador Haar: {cascade_path}")

    carpeta_salida = Path(args.salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    contador = 0

    print(f"Detectando rostros. Guardando imágenes en: {carpeta_salida.resolve()}")
    print("Presiona Esc para terminar.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            rostros = detector.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5)
            for (x, y, ancho, alto) in rostros:
                recorte = frame[y : y + alto, x : x + ancho]
                if recorte.size == 0:
                    continue
                recorte = cv.resize(recorte, (100, 100))
                contador += 1
                destino = carpeta_salida / f"rostro_{contador:04d}.jpg"
                if not cv.imwrite(str(destino), recorte):
                    print(f"No se pudo guardar: {destino}")
                cv.rectangle(frame, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)

            cv.imshow("Reconocimiento facial - Esc para salir", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

    print(f"Listo. Se guardaron {contador} recorte(s) en {carpeta_salida.resolve()}.")


if __name__ == "__main__":
    main()
