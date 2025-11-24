import cv2
import os
import sys

# Acceso al src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detectar_regiones import detectar_carta, corregir_perspectiva, detectar_rotacion, extraer_regiones


# ================================
# CONFIGURAR CARPETAS
# ================================
BASE_NUM = "plantillas/numeros"
BASE_PALO = "plantillas/palos"

os.makedirs(BASE_NUM, exist_ok=True)
os.makedirs(BASE_PALO, exist_ok=True)


# ================================
# CÁMARA
# ================================
cam = cv2.VideoCapture(1)

if not cam.isOpened():
    print("❌ Error: no se pudo abrir la cámara")
    exit()

print("\n🎯 MODO CAPTURA LISTO")
print("Pulsa:")
print("   N → Capturar número")
print("   P → Capturar palo")
print("   Q → Salir\n")

contador_n = 0
contador_p = 0


# ================================
# LOOP PRINCIPAL
# ================================
while True:
    ret, frame = cam.read()
    if not ret:
        continue

    cont, carta = detectar_carta(frame)

    if carta is not None:
        # Corregir perspectiva y orientación
        carta_w = corregir_perspectiva(carta)
        rot = detectar_rotacion(carta_w)

        if rot == 90:
            carta_w = cv2.rotate(carta_w, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 270:
            carta_w = cv2.rotate(carta_w, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Extraer número y palo
        region_num, region_palo = extraer_regiones(carta_w)

        cv2.imshow("Carta corregida", carta_w)
        cv2.imshow("Numero detectado", region_num)
        cv2.imshow("Palo detectado", region_palo)

    cv2.imshow("Camara", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Guardar número
    if key == ord('n') and carta is not None:
        nombre = f"{BASE_NUM}/num_{contador_n}.jpg"
        cv2.imwrite(nombre, region_num)
        print(f"💾 Número guardado: {nombre}")
        contador_n += 1

    # Guardar palo
    if key == ord('p') and carta is not None:
        nombre = f"{BASE_PALO}/palo_{contador_p}.jpg"
        cv2.imwrite(nombre, region_palo)
        print(f"💾 Palo guardado: {nombre}")
        contador_p += 1


cam.release()
cv2.destroyAllWindows()

