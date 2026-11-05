# Detector de rostros

Este programa detecta caras en la cámara web o en un video y guarda cada recorte como JPG de 100 × 100 píxeles. Usa un clasificador Haar incluido con OpenCV; no reconoce la identidad de las personas.

## Instalación en Windows (PowerShell)

Abre PowerShell en esta carpeta (`reconocimiento_cara`) y ejecuta:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno, puedes omitir la línea `Activate.ps1` y ejecutar pip y el programa con su ruta completa:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\reconocimientoFacial.py
```

Si `py` tampoco se reconoce, instala Python 3 desde [python.org](https://www.python.org/downloads/) y marca **Add python.exe to PATH** durante la instalación. Después abre una nueva ventana de PowerShell.

## Uso

Cámara web predeterminada (índice 0):

```powershell
python .\reconocimientoFacial.py
```

Otro índice de cámara:

```powershell
python .\reconocimientoFacial.py 1
```

Video desde un archivo:

```powershell
python .\reconocimientoFacial.py .\caras.mp4
```

Por defecto las imágenes se guardan en `rostros`. Para elegir otra carpeta:

```powershell
python .\reconocimientoFacial.py .\caras.mp4 --salida .\caras_detectadas
```

Presiona **Esc** en la ventana del video para terminar. El detector funciona cuadro por cuadro, así que puede guardar varias imágenes muy parecidas si una cara aparece en muchos cuadros.
