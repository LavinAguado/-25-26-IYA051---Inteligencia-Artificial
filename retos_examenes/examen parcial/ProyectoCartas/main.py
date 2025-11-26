import cv2
from src.deteccion import detectar_carta
from src.detectar_regiones import corregir_perspectiva, extraer_regiones, enderezar_por_pca
from src.plantillas import cargar_plantillas, comparar

def main():
    plantillas_numeros, plantillas_palos = cargar_plantillas()

    cam = cv2.VideoCapture(1)

    if not cam.isOpened():
        print("❌ No se pudo abrir la cámara")
        return

    print("\n📸 Reconocimiento activo — Q para salir\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        pts, carta = detectar_carta(frame)

        if pts is not None and carta is not None:
            
            # --- 1. Perspectiva corregida ---
            warp = corregir_perspectiva(frame, pts)

            # --- 3. Extraer regiones ---
            region_num, region_palo = extraer_regiones(warp)

            # --- 4. Reconocimiento ---
            numero, score_n = comparar(region_num, plantillas_numeros)
            palo, score_p = comparar(region_palo, plantillas_palos)

            texto = f"{numero} de {palo}"
            cv2.putText(frame, texto, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

            cv2.imshow("Carta corregida", warp)
            cv2.imshow("Numero", region_num)
            cv2.imshow("Palo", region_palo)

        cv2.imshow("Camara", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
