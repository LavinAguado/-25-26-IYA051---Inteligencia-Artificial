import cv2
import numpy as np

def detectar_carta(frame):
    """
    Detecta la carta más grande del frame.
    Devuelve (contorno, recorte_carta).
    """
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5,5), 0)
    bordes = cv2.Canny(blur, 40, 120)

    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, None

    contorno = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(contorno)
    if area < 8000:
        return None, None

    x, y, w, h = cv2.boundingRect(contorno)
    carta = frame[y:y+h, x:x+w]

    return contorno, carta
