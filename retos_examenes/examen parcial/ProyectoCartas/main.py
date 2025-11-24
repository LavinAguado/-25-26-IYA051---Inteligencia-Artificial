import cv2
from src.deteccion import detectar_carta
from src.detectar_regiones import corregir_perspectiva, detectar_rotacion, extraer_regiones, detectar_color
from src.plantillas import cargar_plantillas, comparar

def main():
    nums, palos = cargar_plantillas()
    cam = cv2.VideoCapture(1)

    if not cam.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cont, carta = detectar_carta(frame)

        if carta is not None:
            warp = corregir_perspectiva(carta)

            rot = detectar_rotacion(warp)
            if rot == 90:
                warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                warp = cv2.rotate(warp, cv2.ROTATE_180)
            elif rot == 270:
                warp = cv2.rotate(warp, cv2.ROTATE_90_COUNTERCLOCKWISE)

            num_roi, palo_roi = extraer_regiones(warp)

            n, cn = comparar(num_roi, nums)
            p, cp = comparar(palo_roi, palos)

            color = detectar_color(palo_roi)

            if color == "rojo":
                posibles = ["corazon", "diamante"]
            elif color == "negro":
                posibles = ["pica", "trebol"]
            else:
                posibles = list(palos.keys())

            if p not in posibles:
                p = "?"

            texto = f"{n} de {p}"
            cv2.putText(frame, texto, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Carta corregida", warp)
            cv2.imshow("Num", num_roi)
            cv2.imshow("Palo", palo_roi)

        cv2.imshow("Camara", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
