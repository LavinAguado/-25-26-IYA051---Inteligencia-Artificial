import cv2
import numpy as np

def detectar_carta(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5,5), 0)
    bordes = cv2.Canny(blur, 50, 150)

    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contornos) == 0:
        return None, None

    cont = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(cont)

    if area < 5000:  # ajustable
        return None, None

    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)

    if len(approx) != 4:
        # Si no devuelve 4 puntos, NO hacemos warp
        x,y,w,h = cv2.boundingRect(cont)
        return cont.reshape(-1,2).astype("float32"), frame[y:y+h, x:x+w]

    pts = approx.reshape(4,2).astype("float32")

    x,y,w,h = cv2.boundingRect(cont)
    carta = frame[y:y+h, x:x+w]

    return pts, carta
