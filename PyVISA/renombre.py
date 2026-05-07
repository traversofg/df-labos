import os
import re

# Ruta al directorio con los archivos
directorio = r"C:\Users\Fede\OneDrive - UBA\Facultá\2025 1C - Labo 3\NoLineales\Media onda"  # Cambiar si están en otra carpeta

# Listar todos los archivos en ese directorio
for nombre in os.listdir(directorio):
    # Ignorar archivos que no sean CSV o que no sigan el patrón
    if not nombre.endswith(".csv"):
        continue

    # Buscar patrón del tipo RXXXXXX.0
    match = re.search(r"R(\d+)\.0\.csv", nombre)
    if match:
        valor = int(match.group(1))

        # Elegir sufijo adecuado
        if valor >= 1_000_000 and valor % 1_000_000 == 0:
            nuevo_valor = f"{valor // 1_000_000}M"
        elif valor >= 1_000 and valor % 1_000 == 0:
            nuevo_valor = f"{valor // 1_000}K"
        else:
            nuevo_valor = str(valor)

        # Construir nuevo nombre
        nuevo_nombre = re.sub(r"R\d+\.0\.csv", f"R{nuevo_valor}.csv", nombre)

        # Ruta completa
        origen = os.path.join(directorio, nombre)
        destino = os.path.join(directorio, nuevo_nombre)

        # Renombrar
        os.rename(origen, destino)
        print(f"Renombrado: {nombre} → {nuevo_nombre}")
