import cv2
import numpy as np

# -------------------------------------------------------------
# 1. Warp perspectiva
# -------------------------------------------------------------
def corregir_perspectiva(carta):
    gris = cv2.cvtColor(carta, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 40, 120)

    conts, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts:
        return carta

    contorno = max(conts, key=cv2.contourArea)
    peri = cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, 0.015 * peri, True)

    if len(approx) != 4:
        return carta

    pts = approx.reshape(4,2).astype("float32")

    # ordenar puntos
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    destino = np.array([[0,0],[200,0],[0,300],[200,300]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.array([tl,tr,bl,br]), destino)
    warp = cv2.warpPerspective(carta, M, (200,300))

    return warp

# -------------------------------------------------------------
# 2. Detectar rotación en 0 / 90 / 180 / 270
# -------------------------------------------------------------
def detectar_rotacion(carta):
    h, w, _ = carta.shape

    # Tomamos un ROI donde debería estar el número superior
    roi = carta[int(h*0.03):int(h*0.20), int(w*0.03):int(w*0.25)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    tinta = np.mean(gray < 180)

    if tinta < 0.05:
        return 180  # probablemente está boca abajo

    # Detectar si está girada 90°
    if roi.shape[1] > roi.shape[0] * 1.3:
        return 90
    if roi.shape[0] > roi.shape[1] * 1.3:
        return 270

    return 0

# -------------------------------------------------------------
# 3. Extraer número y palo
# -------------------------------------------------------------
def extraer_regiones(carta):
    """
    ROIs definidos para la baraja nueva.
    MAÑANA los ajustamos con tu cámara REAL.
    """
    h, w, _ = carta.shape

    # Número arriba izquierda
    num = carta[int(h*0.02):int(h*0.15), int(w*0.03):int(w*0.18)]

    # Palo pequeño debajo
    palo = carta[int(h*0.15):int(h*0.28), int(w*0.03):int(w*0.18)]

    return num, palo

# -------------------------------------------------------------
# 4. Analizar color del palo
# -------------------------------------------------------------
def detectar_color(region):
    if region is None or region.size == 0:
        return "indefinido"

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0,60,40), (10,255,255))
    red2 = cv2.inRange(hsv, (160,60,40), (179,255,255))
    mask_red = red1 | red2

    value = hsv[:,:,2]
    mask_black = (value < 80)

    if mask_red.mean() > 0.08:
        return "rojo"
    if mask_black.mean() > 0.10:
        return "negro"

    return "indefinido"
