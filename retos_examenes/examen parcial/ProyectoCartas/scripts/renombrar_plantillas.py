import os

# ========================
# CONFIGURACIÓN
# ========================

ruta_numeros = "../plantillas/numeros"
ruta_palos = "../plantillas/palos"

# Orden correcto de una baraja
orden_numeros = [
    "A", "2", "3", "4", "5", "6",
    "7", "8", "9", "10", "J", "Q", "K"
]

orden_palos = [
    "corazon",   # ♥
    "diamante",  # ♦
    "trebol",    # ♣
    "pica"       # ♠
]

# ========================
# RENOMBRAR NÚMEROS
# ========================
def renombrar_numeros():
    print("\n🔢 Renombrando números...")

    archivos = sorted(os.listdir(ruta_numeros))

    if len(archivos) != len(orden_numeros):
        print(f"⚠ Atención: hay {len(archivos)} archivos, pero deberían ser {len(orden_numeros)}.")
        print("Asegúrate de que capturaste exactamente A,2,3,...K en ese orden.")
        return

    for i, archivo in enumerate(archivos):
        nombre_final = orden_numeros[i] + ".jpg"
        origen = os.path.join(ruta_numeros, archivo)
        destino = os.path.join(ruta_numeros, nombre_final)
        os.rename(origen, destino)
        print(f"✔ {archivo} → {nombre_final}")

    print("✅ Números renombrados correctamente.")


# ========================
# RENOMBRAR PALOS
# ========================
def renombrar_palos():
    print("\n♣﻿♦﻿♥﻿♠ Renombrando palos...")

    archivos = sorted(os.listdir(ruta_palos))

    if len(archivos) != len(orden_palos):
        print(f"⚠ Atención: hay {len(archivos)} archivos, pero deberían ser 4.")
        print("Asegúrate de haber capturado CORAZÓN → DIAMANTE → TRÉBOL → PICA en ese orden.")
        return

    for i, archivo in enumerate(archivos):
        nombre_final = orden_palos[i] + ".jpg"
        origen = os.path.join(ruta_palos, archivo)
        destino = os.path.join(ruta_palos, nombre_final)
        os.rename(origen, destino)
        print(f"✔ {archivo} → {nombre_final}")

    print("✅ Palos renombrados correctamente.")


# ========================
# EJECUCIÓN
# ========================
if __name__ == "__main__":
    renombrar_numeros()
    renombrar_palos()
    print("\n🎉 TODO RENOMBRADO PERFECTAMENTE\n")
