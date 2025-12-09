import cv2
import numpy as np

import re
import math
import shutil
import os

IMG_FOLDER = "/home/pablo/Documents/pro_vision/planos/comprobar_manual/images/10.jpg"

img = cv2.imread(IMG_FOLDER)

# tolerancia zona superior donde buscar la linea (ej: 20% superior)
TOP_CROP_RATIO = 0.50

# color rango HSV para la linea gris-verdosa (ajustable)
HSV_MIN = np.array([0, 0, 0])   # hue, sat, val
HSV_MAX = np.array([95, 200, 255])

# Hough params
HOUGH_RHO = 1
HOUGH_THETA = np.pi/180
HOUGH_THRESHOLD = 50

def detect_scale_line_and_number(img):
    """
    Intenta detectar la línea de escala y el número encima.
    Devuelve: (line_px_length, number_mm) o (None, None) si falla.
    """
    H, W = img.shape[:2]
    top_h = int(H * TOP_CROP_RATIO)
    top_region = img[0:top_h, :].copy()

    # convertir a HSV y enmascarar color aproximado
    hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_MIN, HSV_MAX)
    # limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # bordes y Hough
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    min_len = int(W * 0.30)
    lines = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, minLineLength=min_len, maxLineGap=50)
    if lines is None:
        return None, None

    # elegir la línea más horizontal y más larga (cercana a la parte superior)
    best_line = None
    best_len = 0
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = math.hypot(x2 - x1, y2 - y1)
        angle_deg = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        # casi horizontal: ángulo < 10 grados
        if angle_deg < 10 and length > best_len:
            best_len = length
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        return None, None

    x1, y1, x2, y2 = best_line
    # longitud en px (en la imagen completa, no sólo recorte)
    # ajustar coordenadas a imagen completa (top_region origin 0)
    line_len_px = math.hypot(x2 - x1, y2 - y1)

    # ROI para OCR: caja encima de la línea (mayor seguridad)
    # coordenadas en la imagen global:
    y_line_global = y2  # dentro top_region
    # define top box
    roi_y1 = max(0, y_line_global - int(top_h * 0.4) - 10)
    roi_y2 = max(0, y_line_global - 2)
    roi_x1 = max(0, int(min(x1, x2) - 0.05 * W))
    roi_x2 = min(W, int(max(x1, x2) + 0.05 * W))
    roi = img[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    if roi.size == 0:
        return line_len_px, None

    # preprocesado OCR: convertir a gris y binarizar
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # aumentar contraste
    gray = cv2.equalizeHist(gray)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # usar pytesseract para leer solo dígitos
    try:
        ensure_tesseract()
        custom_oem_psm_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thr, config=custom_oem_psm_config)
    except Exception as e:
        # en caso de que tesseract no esté disponible, devolvemos None para que haga fallback manual
        print("[AVISO] OCR no disponible o falló:", e)
        return line_len_px, None

    # extraer dígitos
    digits = re.findall(r'\d+', text)
    if not digits:
        return line_len_px, None
    # tomar el mayor conjunto de dígitos leídos (por si hay fragmentos)
    number_str = max(digits, key=len)
    try:
        number_mm = int(number_str)
        return line_len_px, number_mm
    except:
        return line_len_px, None
    

line_len_px, number_mm = detect_scale_line_and_number(img)

print ("Longitud línea escala (px): %s\n" % (line_len_px if line_len_px is not None else "No detectada"))