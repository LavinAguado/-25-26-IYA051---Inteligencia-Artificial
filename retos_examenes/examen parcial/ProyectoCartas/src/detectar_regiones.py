import cv2
import numpy as np

# -------------------------------------------------------------
# 1. Warp perspectiva (recibe frame y puntos)
# -------------------------------------------------------------
def corregir_perspectiva(frame, pts):
    import numpy as np
    import cv2

    pts = np.array(pts, dtype="float32")

    # Ordenar puntos correctamente (TL, TR, BR, BL)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Tamaño estándar para cartas Fournier (alto>ancho)
    W = 200
    H = 280   # proporción 1.40

    dst = np.array([[0, 0],
                    [W - 1, 0],
                    [W - 1, H - 1],
                    [0, H - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
    warp = cv2.warpPerspective(frame, M, (W, H))

    return warp

# -------------------------------------------------------------
# 2. Rotación simple y estable (0 o 180 grados)
# -------------------------------------------------------------
def detectar_rotacion(carta):
    h, w, _ = carta.shape
    roi = carta[int(h*0.02):int(h*0.18), int(w*0.02):int(w*0.25)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # tinta arriba -> normal, tinta abajo -> al revés
    tinta = np.mean(gray < 180)

    if tinta < 0.03:
        return 180
    return 0


# -------------------------------------------------------------
# 3. Regiones calibradas para aluminio Fournier blancas
# -------------------------------------------------------------
def extraer_regiones(carta):
    """
    Recorta las dos zonas donde SIEMPRE hay número y palo
    en una carta estándar (88x63 mm), ya escalada a 200x300.
    """

    h, w, _ = carta.shape

    # --- Región del número (arriba izquierda) ---
    # Número: muy arriba y muy a la izquierda
    num_y1 = int(h * 0.02)
    num_y2 = int(h * 0.15)
    num_x1 = int(w * 0.02)
    num_x2 = int(w * 0.18)

    region_num = carta[num_y1:num_y2, num_x1:num_x2]

    # --- Región del palo (justo debajo del número) ---
    palo_y1 = int(h * 0.15)
    palo_y2 = int(h * 0.28)
    palo_x1 = int(w * 0.02)
    palo_x2 = int(w * 0.18)

    region_palo = carta[palo_y1:palo_y2, palo_x1:palo_x2]

    return region_num, region_palo


# ======================================================
# 5) DETECTAR COLOR DEL PALO
# ======================================================
def detectar_color(region):
    if region is None or region.size == 0:
        return "indefinido"

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # rojo
    red1 = cv2.inRange(hsv, (0, 60, 40), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 60, 40), (179, 255, 255))
    mask_red = red1 | red2

    # negro
    value = hsv[:, :, 2]
    mask_black = (value < 80)

    if mask_red.mean() > 0.08:
        return "rojo"
    if mask_black.mean() > 0.10:
        return "negro"

    return "indefinido"
