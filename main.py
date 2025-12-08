import zipfile
import os

# Nombre del archivo zip (según tu imagen)
archivo_zip = "credit-risk-mlops.zip"

# Directorio donde quieres extraerlo (opcional, aquí se crea una carpeta con el mismo nombre)
directorio_destino = "credit-risk-mlops-extracted"

# Crear la carpeta de destino si no existe
os.makedirs(directorio_destino, exist_ok=True)

try:
    with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
        zip_ref.extractall(directorio_destino)
    print(f"Archivo descomprimido exitosamente en: {directorio_destino}")
except FileNotFoundError:
    print(f"No se encontró el archivo: {archivo_zip}")