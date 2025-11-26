import cv2
import os
import sys

# Acceder al módulo src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.deteccion import detectar_carta
from src.detectar_regiones import corregir_perspectiva, detectar_rotacion, extraer_regiones

BASE_NUM = "plantillas/numeros"
BASE_PALO = "plantillas/palos"

os.makedirs(BASE_NUM, exist_ok=True)
os.makedirs(BASE_PALO, exist_ok=True)

cam = cv2.VideoCapture(1)

print("\n🎯 CAPTURADOR DE PLANTILLAS")
print("N = guardar número")
print("P = guardar palo")
print("Q = salir\n")

contador_n = 0
contador_p = 0

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    pts, carta = detectar_carta(frame)

    if pts is not None:
        warp = corregir_perspectiva(frame, pts)
        rot = detectar_rotacion(warp)

        if rot == 90:
            warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            warp = cv2.rotate(warp, cv2.ROTATE_180)
        elif rot == 270:
            warp = cv2.rotate(warp, cv2.ROTATE_90_COUNTERCLOCKWISE)

        region_num, region_palo = extraer_regiones(warp)

        cv2.imshow("Carta corregida", warp)
        cv2.imshow("Numero detectado", region_num)
        cv2.imshow("Palo detectado", region_palo)

    cv2.imshow("Camara", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('n') and pts is not None:
        nombre = f"{BASE_NUM}/num_{contador_n}.jpg"
        cv2.imwrite(nombre, region_num)
        print("👉 Guardado:", nombre)
        contador_n += 1

    if key == ord('p') and pts is not None:
        nombre = f"{BASE_PALO}/palo_{contador_p}.jpg"
        cv2.imwrite(nombre, region_palo)
        print("👉 Guardado:", nombre)
        contador_p += 1

cam.release()
cv2.destroyAllWindows()
