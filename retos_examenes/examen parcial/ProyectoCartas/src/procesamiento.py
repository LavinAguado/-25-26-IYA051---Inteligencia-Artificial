import cv2
import numpy as np

def corregir_perspectiva(imagen):
    """
    Corrige la perspectiva de la carta para que quede rectangular y alineada.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5,5), 0)
    bordes = cv2.Canny(blur, 50, 150)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contornos) == 0:
        return imagen  # Si no hay contornos, devolvemos la imagen original

    # Contorno más grande
    contorno = max(contornos, key=cv2.contourArea)
    peri = cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, 0.02 * peri, True)

    if len(approx) != 4:
        return imagen  # No podemos corregir si no es cuadrilátero

    # Ordenar los puntos (tl, tr, br, bl)
    pts = approx.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.array([
        pts[np.argmin(s)],     # top-left
        pts[np.argmin(diff)],  # top-right
        pts[np.argmax(s)],     # bottom-right
        pts[np.argmax(diff)]   # bottom-left
    ], dtype="float32")

    # Medidas finales de la carta
    ancho = max(np.linalg.norm(rect[0]-rect[1]), np.linalg.norm(rect[2]-rect[3]))
    alto  = max(np.linalg.norm(rect[0]-rect[3]), np.linalg.norm(rect[1]-rect[2]))

    dst = np.array([
        [0,0],
        [ancho-1,0],
        [ancho-1,alto-1],
        [0,alto-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(imagen, M, (int(ancho), int(alto)))

    return warp
