import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.deteccion import detectar_carta
from src.detectar_regiones import corregir_perspectiva, detectar_rotacion, extraer_regiones

BASE_NUM = "plantillas/numeros"
BASE_PALO = "plantillas/palos"

os.makedirs(BASE_NUM, exist_ok=True)
os.makedirs(BASE_PALO, exist_ok=True)

cam = cv2.VideoCapture(1)

if not cam.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("\n🎯 CAPTURA LISTA — 'N' guarda número, 'P' guarda palo, 'Q' sale\n")

cN = 0
cP = 0

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    pts, _ = detectar_carta(frame)

    if pts is not None:

        warp = corregir_perspectiva(frame, pts)

        rot = detectar_rotacion(warp)
        if rot == 180:
            warp = cv2.rotate(warp, cv2.ROTATE_180)

        num, palo = extraer_regiones(warp)

        cv2.imshow("Carta corregida", warp)
        cv2.imshow("Número", num)
        cv2.imshow("Palo", palo)

    cv2.imshow("Camara", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if pts is not None:

        if key == ord('n'):
            name = f"{BASE_NUM}/num_{cN}.jpg"
            cv2.imwrite(name, num)
            print("💾 Guardado número:", name)
            cN += 1

        if key == ord('p'):
            name = f"{BASE_PALO}/palo_{cP}.jpg"
            cv2.imwrite(name, palo)
            print("💾 Guardado palo:", name)
            cP += 1

cam.release()
cv2.destroyAllWindows()
