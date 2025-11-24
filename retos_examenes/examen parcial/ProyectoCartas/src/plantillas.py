import cv2
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "plantillas")

def cargar_plantillas():
    nums = {}
    palos = {}

    for tipo, dicc in [("numeros", nums), ("palos", palos)]:
        carpeta = os.path.join(BASE, tipo)
        if not os.path.exists(carpeta):
            continue
        for f in os.listdir(carpeta):
            if f.lower().endswith((".jpg",".png",".jpeg")):
                img = cv2.imread(os.path.join(carpeta,f), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (80,100))
                dicc[os.path.splitext(f)[0]] = img

    return nums, palos

def comparar(region, plantillas):
    if region is None or region.size == 0:
        return None, 0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (80,100))

    best = None
    best_val = -1

    for nombre, plantilla in plantillas.items():
        res = cv2.matchTemplate(gray, plantilla, cv2.TM_CCOEFF_NORMED)
        _, val, _, _ = cv2.minMaxLoc(res)

        if val > best_val:
            best_val = val
            best = nombre

    return best, best_val
