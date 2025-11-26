import cv2
from src.deteccion import detectar_carta
from src.detectar_regiones import corregir_perspectiva, extraer_regiones
from src.plantillas import cargar_plantillas, comparar

def main():
    # Cargar plantillas
    plantillas_numeros, plantillas_palos = cargar_plantillas()

    # Usar la cámara principal de DroidCam
    cam = cv2.VideoCapture(1)

    if not cam.isOpened():
        print("❌ No se pudo abrir la cámara")
        return

    print("📸 Reconocimiento listo. Pulsa Q para salir.\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        # 1. Detectar carta
        pts, carta = detectar_carta(frame)

        if carta is not None:
            # 2. Corregir perspectiva (warp)
            warp = corregir_perspectiva(carta)

            # 3. Extraer número y palo
            region_num, region_palo = extraer_regiones(warp)

            # 4. Reconocimiento por plantillas
            numero, score_n = comparar(region_num, plantillas_numeros)
            palo, score_p = comparar(region_palo, plantillas_palos)

            # 5. Mostrar resultado
            texto = f"{numero} de {palo}"
            cv2.putText(frame, texto, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

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
