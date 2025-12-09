#!/usr/bin/env python3
# scripts/test_yolov8_full.py
# Integrado: detección escala automática (línea + OCR) con fallback interactivo (opción A),
# inferencia YOLOv8, agrupado de zonas, colocación de sensores, Excel y PDF.
# [No verificado]

import os
import re
from pathlib import Path
import math
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

# OCR
try:
    import pytesseract
except Exception:
    pytesseract = None

# ----------------------
# CONFIGURACIÓN (ajusta aquí)
# ----------------------
MODEL_PATH = "/home/pablo/Documents/pro_vision/planos/models/yolo_trenes/weights/best.pt"
IMG_FOLDER = "/home/pablo/Documents/pro_vision/planos/comprobar_manual/images"
OUTPUT_FOLDER = "/home/pablo/Documents/pro_vision/planos/comprobar_manual/results_with_sensors"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# tolerancia zona superior donde buscar la linea (ej: 20% superior)
TOP_CROP_RATIO = 0.20

# color rango HSV para la linea gris-verdosa (ajustable)
HSV_MIN = np.array([35, 20, 40])   # hue, sat, val
HSV_MAX = np.array([95, 200, 255])

# Hough params
HOUGH_RHO = 1
HOUGH_THETA = np.pi/180
HOUGH_THRESHOLD = 50

# Gap para unir bboxes consecutivas (metros)
MERGE_GAP_M = 0.10  # 10 cm

# clases (debe coincidir con tu modelo)
CLASS_NAMES = [
    "cabina",
    "salon",
    "vestibulo",
    "wc_normal",
    "bufet",
    "fuelles",
    "anexo",
    "bicicletas",
    "personal",
    "wc_pmr",
    "corredor"
]

# Colores BGR + alpha para dibujo
CLASS_COLORS = {
    "cabina": (0, 100, 0, 0.30),
    "salon": (100, 0, 0, 0.30),
    "vestibulo": (0, 120, 255, 0.30),
    "wc_normal": (0, 0, 150, 0.30),
    "bufet": (0, 180, 180, 0.30),
    "fuelles": (150, 0, 150, 0.30),
    "anexo": (80, 80, 80, 0.30),
    "bicicletas": (200, 80, 200, 0.30),
    "personal": (100, 50, 0, 0.30),
    "wc_pmr": (0, 150, 80, 0.30),
    "corredor": (255, 150, 0, 0.30)
}

# Alturas (m) según documento (category R example)
HEIGHTS = {
    "seated": [0.10, 0.60, 1.00, 1.10],
    "standing": [0.10, 1.10, 1.70],
    "rh_co2": 1.10,
    "floor": 0.0
}

# Marker colors for sensor types (for PDF legend)
MARKER_COLORS = {
    "AT": "orange",
    "AT_standing": "red",
    "RH": "cyan",
    "CO2": "magenta",
    "TS_Fl": "green"
}

# ----------------------
# UTILIDADES
# ----------------------
def px_to_m(pixels, mm_per_px):
    mm = pixels * mm_per_px
    return mm / 1000.0

def m_to_px(meters, mm_per_px):
    mm = meters * 1000.0
    return int(round(mm / mm_per_px))

def ensure_tesseract():
    if pytesseract is None:
        raise RuntimeError("pytesseract no está instalado. Instálalo: pip install pytesseract")
    # also system tesseract
    try:
        _ = pytesseract.get_tesseract_version()
    except Exception:
        raise RuntimeError("Tesseract OCR no está instalado en el sistema. En Ubuntu: sudo apt install tesseract-ocr")

# ----------------------
# DETECCIÓN DE LA LÍNEA DE ESCALA + OCR
# ----------------------
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
    

# ----------------------
# SELECCIÓN MANUAL DE LA LÍNEA DE ESCALA SI FALLA LA DETECCIÓN
# ----------------------

manual_points = []

def click_event_scale(event, x, y, flags, param):
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN:
        manual_points.append((x, y))
        print(f" → Punto marcado: {x}, {y}")


def get_scale_manual(img, filename):
    """
    Modo manual: abre la imagen, el usuario marca dos puntos que representan la línea de escala.
    Después pide el número en mm desde terminal.
    Devuelve (line_len_px, number_mm)
    """
    global manual_points
    manual_points = []

    clone = img.copy()
    cv2.namedWindow("Selecciona los extremos de la escala")
    cv2.setMouseCallback("Selecciona los extremos de la escala", click_event_scale)

    print("\n[ MODO MANUAL ACTIVADO ]")
    print(f"Imagen: {filename}")
    print("Instrucciones:")
    print("  1. Haz CLICK IZQUIERDO en el primer extremo de la línea de escala.")
    print("  2. Haz CLICK IZQUIERDO en el segundo extremo.")
    print("  3. Después cierra la ventana para continuar.\n")

    while True:
        temp = clone.copy()
        for p in manual_points:
            cv2.circle(temp, p, 5, (0, 0, 255), -1)
        cv2.imshow("Selecciona los extremos de la escala", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para abortar
            cv2.destroyWindow("Selecciona los extremos de la escala")
            return None, None
        if len(manual_points) == 2:
            break

    cv2.destroyWindow("Selecciona los extremos de la escala")

    # cálculo de la distancia
    (x1, y1), (x2, y2) = manual_points
    line_len_px = math.hypot(x2 - x1, y2 - y1)
    print(f"\nDistancia en píxeles: {line_len_px:.2f}")

    # pedir número real
    while True:
        user_val = input(f"Introduce el número de la escala en mm para {filename}: ").strip()
        try:
            number_mm = int(re.sub(r'\D', '', user_val))
            break
        except:
            print("Valor inválido, introduce un entero en mm.")

    return line_len_px, number_mm



# ----------------------
# AGRUPADO Y SENSORES (similares a versión previa)
# ----------------------
def group_consecutive_boxes(boxes_px, mm_per_px, merge_gap_m=MERGE_GAP_M):
    """
    boxes_px: list of (x1, y1, x2, y2, cls_name, conf)
    return: dict class -> list of merged zones (xL_px, xR_px, boxes_list)
    """
    result = {}
    for cls in CLASS_NAMES:
        cls_boxes = [b for b in boxes_px if b[4] == cls]
        if not cls_boxes:
            result[cls] = []
            continue
        cls_boxes.sort(key=lambda b: b[0])
        merged = []
        cur_left, cur_right, cur_list = cls_boxes[0][0], cls_boxes[0][2], [cls_boxes[0]]
        for b in cls_boxes[1:]:
            gap_px = b[0] - cur_right
            gap_m = px_to_m(max(0, gap_px), mm_per_px)
            if gap_m <= merge_gap_m:
                cur_right = max(cur_right, b[2])
                cur_list.append(b)
            else:
                merged.append((cur_left, cur_right, cur_list))
                cur_left, cur_right, cur_list = b[0], b[2], [b]
        merged.append((cur_left, cur_right, cur_list))
        result[cls] = merged
    return result

def decide_subzones(L_m):
    if L_m < 5.0:
        return 1
    elif 5.0 <= L_m < 10.0:
        return 2
    else:
        return 3

def place_sensors_for_subzone(zone_name, left_m, right_m, subzone_idx, total_subzones, zone_global_idx):
    sensors = []
    width_m = right_m - left_m
    quarter_x = left_m + 0.25 * width_m
    # seated: 4 heights at quarter_x
    for i, h in enumerate(HEIGHTS["seated"], start=1):
        name = f"S{zone_global_idx}.{subzone_idx}_AT_{i:02d}"
        sensors.append({
            "name": name, "type": "AT", "zone": zone_name,
            "subzone": f"{zone_global_idx}.{subzone_idx}", "x_m": quarter_x, "height_m": h
        })
    # standing at center
    cx = left_m + 0.5 * width_m
    for i, h in enumerate(HEIGHTS["standing"], start=1):
        name = f"S{zone_global_idx}.{subzone_idx}_AT_S{i:02d}"
        sensors.append({
            "name": name, "type": "AT_standing", "zone": zone_name,
            "subzone": f"{zone_global_idx}.{subzone_idx}", "x_m": cx, "height_m": h
        })
    # RH and CO2 and floor center
    sensors.append({
        "name": f"S{zone_global_idx}.{subzone_idx}_RH_01", "type": "RH", "zone": zone_name,
        "subzone": f"{zone_global_idx}.{subzone_idx}", "x_m": cx, "height_m": HEIGHTS["rh_co2"]
    })
    sensors.append({
        "name": f"S{zone_global_idx}.{subzone_idx}_CO2_01", "type": "CO2", "zone": zone_name,
        "subzone": f"{zone_global_idx}.{subzone_idx}", "x_m": cx, "height_m": HEIGHTS["rh_co2"]
    })
    sensors.append({
        "name": f"S{zone_global_idx}.{subzone_idx}_TS_FL_01", "type": "TS_Fl", "zone": zone_name,
        "subzone": f"{zone_global_idx}.{subzone_idx}", "x_m": cx, "height_m": HEIGHTS["floor"]
    })
    return sensors

# ----------------------
# DIBUJO helpers
# ----------------------
def draw_filled_rect(img, x1, y1, x2, y2, color):
    overlay = img.copy()
    bgr = tuple(map(int, color[:3]))
    alpha = color[3]
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 2)

# ----------------------
# CARGA MODELO
# ----------------------
print("Cargando modelo:", MODEL_PATH)
model = YOLO(MODEL_PATH)

# ----------------------
# PROCESO PRINCIPAL
# ----------------------
image_files = sorted([p for p in Path(IMG_FOLDER).glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
if not image_files:
    print("No hay imágenes en:", IMG_FOLDER)
    sys.exit(1)

for img_path_obj in image_files:
    img_path = str(img_path_obj)
    print("\n=== Procesando:", img_path_obj.name, "===")
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    # 1) detectar escala
    line_len_px, number_mm = detect_scale_line_and_number(img)


    # Si falla la detección, activar modo manual
    if line_len_px is None or number_mm is None:
        print(f"[AVISO] Detección automática fallida en {img_path_obj.name}. Activando modo manual.")
        line_len_px, number_mm = get_scale_manual(img, img_path_obj.name)

    if line_len_px is None or number_mm is None:
        print("[AVISO] No se obtuvo la escala. Saltando imagen.")
        continue


    mm_per_px = number_mm / line_len_px
    print(f"Escala detectada: {number_mm} mm en {line_len_px:.1f} px -> mm/px = {mm_per_px:.4f}")

    # 2) inferencia YOLO
    results = model.predict(source=img_path, conf=0.25, show=False)

    boxes_px = []
    for r in results:
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            cls_id = int(cls_id)
            conf = float(conf)
            cls_name = model.names[cls_id] if cls_id in model.names else CLASS_NAMES[cls_id]
            boxes_px.append((x1, y1, x2, y2, cls_name, conf))

    if not boxes_px:
        print("No se detectaron objetos en la imagen, se salta.")
        continue

    # 3) global span
    global_xmin = min(b[0] for b in boxes_px)
    global_xmax = max(b[2] for b in boxes_px)
    span_px = global_xmax - global_xmin
    span_m = px_to_m(span_px, mm_per_px)
    print(f"Span total (px): {span_px:.1f} -> {span_m:.3f} m")

    # 4) agrupar por clase
    merged = group_consecutive_boxes(boxes_px, mm_per_px, MERGE_GAP_M)

    # 5) generar sensores
    sensors_list = []
    zone_global_counter = 1
    # comfort zones: salon and bufet
    for cls in ("salon", "bufet"):
        zones = merged.get(cls, [])
        for local_idx, (xL_px, xR_px, boxlist) in enumerate(zones, start=1):
            left_m = px_to_m(xL_px - global_xmin, mm_per_px)
            right_m = px_to_m(xR_px - global_xmin, mm_per_px)
            L_m = right_m - left_m
            n_sub = decide_subzones(L_m)
            sub_w = (right_m - left_m) / n_sub if n_sub > 0 else 0.0
            for s in range(n_sub):
                s_left = left_m + s * sub_w
                s_right = s_left + sub_w
                sensors = place_sensors_for_subzone(cls, s_left, s_right, s+1, n_sub, zone_global_counter)
                sensors_list.extend(sensors)
            zone_global_counter += 1

    # cabina
    cab_zones = merged.get("cabina", [])
    for idx, (xL_px, xR_px, boxlist) in enumerate(cab_zones, start=1):
        left_m = px_to_m(xL_px - global_xmin, mm_per_px)
        right_m = px_to_m(xR_px - global_xmin, mm_per_px)
        cx_m = (left_m + right_m) / 2.0
        sensors_list.append({
            "name": f"Cab{idx}_AT_seated_01", "type": "AT", "zone": "cabina",
            "subzone": f"cab{idx}.1", "x_m": cx_m, "height_m": HEIGHTS["seated"][0]
        })

    # local annexes
    for cls in ("wc_normal", "wc_pmr", "vestibulo", "corredor", "anexo"):
        zones = merged.get(cls, [])
        for idx, (xL_px, xR_px, boxlist) in enumerate(zones, start=1):
            left_m = px_to_m(xL_px - global_xmin, mm_per_px)
            right_m = px_to_m(xR_px - global_xmin, mm_per_px)
            cx_m = (left_m + right_m) / 2.0
            if cls in ("wc_normal", "wc_pmr"):
                sensors_list.append({
                    "name": f"WC{idx}_AT_01", "type": "AT", "zone": cls,
                    "subzone": f"{cls}{idx}.1", "x_m": cx_m, "height_m": 1.1
                })
                sensors_list.append({
                    "name": f"WC{idx}_RH_01", "type": "RH", "zone": cls,
                    "subzone": f"{cls}{idx}.1", "x_m": cx_m, "height_m": HEIGHTS["rh_co2"]
                })
            elif cls == "vestibulo":
                sensors_list.append({
                    "name": f"Vest{idx}_AT_01", "type": "AT", "zone": cls,
                    "subzone": f"{cls}{idx}.1", "x_m": cx_m, "height_m": 0.1
                })
                sensors_list.append({
                    "name": f"Vest{idx}_AT_02", "type": "AT", "zone": cls,
                    "subzone": f"{cls}{idx}.1", "x_m": cx_m, "height_m": 1.7
                })
            else:
                for i, h in enumerate(HEIGHTS["standing"], start=1):
                    sensors_list.append({
                        "name": f"{cls}{idx}_AT_S{i:02d}", "type": "AT_standing", "zone": cls,
                        "subzone": f"{cls}{idx}.1", "x_m": cx_m, "height_m": h
                    })
                sensors_list.append({
                    "name": f"{cls}{idx}_RH_01", "type": "RH", "zone": cls,
                    "subzone": f"{cls}{idx}.1", "x_m": cx_m, "height_m": HEIGHTS["rh_co2"]
                })

    # 6) Guardar Excel
    if sensors_list:
        df = pd.DataFrame(sensors_list)
        excel_path = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}_sensors.xlsx"
        df.to_excel(excel_path, index=False)
        print("Excel guardado en:", excel_path)

    # 7) Dibujar imagen y PDF
    img_vis = img.copy()
    # dibujar zonas (fills)
    for cls, zones in merged.items():
        color = CLASS_COLORS.get(cls, (120,120,120,0.25))
        for (xL_px, xR_px, _) in zones:
            draw_filled_rect(img_vis, xL_px, 0, xR_px, H, color)
    # dibujar bboxes
    for (x1,y1,x2,y2,cls,conf) in boxes_px:
        col = tuple(map(int, CLASS_COLORS.get(cls,(120,120,120,0.25))[:3]))
        cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img_vis, f"{cls} {conf:.2f}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    # preparar figura matplotlib para PDF
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    # dibujar subzona límites (dashed vertical lines) para comfort zones
    for cls in ("salon","bufet"):
        zones = merged.get(cls, [])
        for (xL_px, xR_px, _) in zones:
            left_m = px_to_m(xL_px - global_xmin, mm_per_px)
            right_m = px_to_m(xR_px - global_xmin, mm_per_px)
            L_m = right_m - left_m
            n_sub = decide_subzones(L_m)
            if n_sub > 1:
                sub_w_px = (xR_px - xL_px) / n_sub
                for s in range(1, n_sub):
                    xline = xL_px + s * sub_w_px
                    ax.add_line(Line2D([xline, xline], [0, H], linestyle=(0,(5,5)),
                                       color=np.array(CLASS_COLORS.get(cls,(120,120,120,0.3))[:3])/255.0, linewidth=2))

    # dibujar sensores (estrellas) y leyenda
    for s in sensors_list:
        x_px = global_xmin + m_to_px(s["x_m"], mm_per_px)
        y_px = int(H * 0.5)  # aproximación vertical
        color = MARKER_COLORS.get(s["type"].split('_')[0], "white")
        ax.plot(x_px, y_px, marker='*', markersize=10, color=color)
        ax.text(x_px+3, y_px+3, s["name"], color='white', fontsize=7)

    # leyenda
    legend_handles = [Line2D([0],[0], marker='*', color='w', label=k, markerfacecolor=v, markersize=10)
                      for k,v in MARKER_COLORS.items()]
    ax.legend(handles=legend_handles, loc='lower right', fontsize='small')

    pdf_path = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}_annotated.pdf"
    img_out_path = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}_annotated.jpg"
    fig.savefig(str(pdf_path), bbox_inches='tight', dpi=200)
    plt.close(fig)
    cv2.imwrite(str(img_out_path), img_vis)

    # 8) Guardar detecciones txt normalizado YOLO
    dets_txt = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}.txt"
    with open(dets_txt, "w") as f:
        for (x1,y1,x2,y2,cls,conf) in boxes_px:
            xc = ((x1 + x2)/2)/W
            yc = ((y1 + y2)/2)/H
            bw = (x2 - x1)/W
            bh = (y2 - y1)/H
            cls_id = CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 0
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Guardado: PDF={pdf_path}, IMG={img_out_path}, EXCEL={excel_path}, DETS={dets_txt}")

print("\nTodo el proceso ha terminado. Resultados en:", OUTPUT_FOLDER)
