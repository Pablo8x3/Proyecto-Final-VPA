import cv2
import numpy as np
import pytesseract
import math
import re 
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

# Configuración de ruta (tu ruta original)
IMG_PATH = "/home/pablo/Documents/pro_vision/planos/comprobar_manual/images/10.jpg"

def procesar_plano(img_path):
    # 1. Cargar imagen
    img = cv2.imread(img_path)
    if img is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Preprocesamiento para detectar líneas
    # Aplicar un umbral o Canny para resaltar bordes
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 3. Detectar líneas usando la Transformada de Hough Probabilística
    # minLineLength: Longitud mínima para ser considerada línea (ajústalo según el tamaño de tu imagen)
    # maxLineGap: Brecha máxima entre puntos para considerarlos la misma línea
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=200, maxLineGap=20)

    if lines is None:
        print("No se detectaron líneas.")
        return

    # 4. Filtrar la línea de cota (asumimos que es horizontal y larga)
    best_line = None
    max_length = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calcular longitud
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calcular ángulo para asegurar que sea más o menos horizontal (tolerancia +/- 10 grados)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < 10 or abs(angle) > 170:
            if length > max_length:
                max_length = length
                best_line = (x1, y1, x2, y2)

    if best_line is None:
        print("No se encontró una línea horizontal adecuada.")
        return

    x1, y1, x2, y2 = best_line
    distancia_pixeles = max_length
    print(f"Línea detectada: P1({x1},{y1}) - P2({x2},{y2})")
    print(f"Longitud en píxeles: {distancia_pixeles:.2f} px")

    # Dibujar la línea en la imagen para visualizar
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 5. Detectar y leer el número (OCR)
    # Definimos una Región de Interés (ROI) alrededor del centro de la línea
    # Asumimos que el texto está "encima" o en el centro de la línea
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Definir un margen para buscar el texto (ajusta el tamaño del cuadro según tu imagen)
    w_roi, h_roi = 200, 100 
    y_start = max(0, center_y - h_roi)
    y_end = min(gray.shape[0], center_y + 10) # Un poco por debajo de la línea
    x_start = max(0, center_x - w_roi // 2)
    x_end = min(gray.shape[1], center_x + w_roi // 2)

    roi = gray[y_start:y_end, x_start:x_end]

    # Preprocesar ROI para OCR (Umbralización para dejar texto negro sobre blanco o viceversa)
    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Configuración de Tesseract para buscar solo dígitos
    # psm 6: Asume un bloque de texto uniforme. psm 7: Asume una sola línea de texto.
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    texto = pytesseract.image_to_string(roi_thresh, config=custom_config)

    # Limpiar el texto para obtener solo números
    numeros = re.findall(r'\d+', texto)
    
    if not numeros:
        print("No se pudo leer el número. Intenta ajustar el preprocesamiento del ROI.")
        # Mostrar ROI para depurar
        cv2.imshow("ROI Texto", roi_thresh)
        cv2.waitKey(0)
        return

    # Tomamos el número más grande encontrado (a veces detecta ruido como números pequeños)
    medida_mm = float(max(numeros, key=lambda x: int(x)))
    print(f"Medida leída (OCR): {medida_mm} mm")

    # 6. Calcular factor de conversión
    # Factor: Cuántos metros representa un píxel
    mm_por_pixel = medida_mm / distancia_pixeles
    metros_por_pixel = mm_por_pixel / 1000.0

    print(f"Factor de conversión: {metros_por_pixel:.6f} metros/pixel")
