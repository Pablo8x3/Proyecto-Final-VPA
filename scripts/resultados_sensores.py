#!/usr/bin/env python3
# scripts/test_yolov8_full_fixed.py
# [No verificado]
# Requisitos: pip install ultralytics opencv-python numpy pandas matplotlib openpyxl pytesseract
# Tesseract debe estar instalado en el sistema: sudo apt install tesseract-ocr

import os
import re
import math
from pathlib import Path
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
# CONFIGURACIÓN (ajusta)
# ----------------------
MODEL_PATH = "/home/pablo/Documents/pro_vision/planos/models/yolo_trenes/weights/best.pt"
IMG_FOLDER = "/home/pablo/Documents/pro_vision/planos/comprobar_manual/images"
OUTPUT_FOLDER = "/home/pablo/Documents/pro_vision/planos/comprobar_manual/results_fixed"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TOP_CROP_RATIO = 0.20  # zona superior para buscar la línea
HSV_MIN = np.array([35, 20, 40])
HSV_MAX = np.array([95, 200, 255])

HOUGH_RHO = 1
HOUGH_THETA = np.pi/180
HOUGH_THRESHOLD = 50

MERGE_GAP_M = 0.10  # gap para unir cajas consecutivas (seguimos usando 10cm si fuera necesario)

CLASS_NAMES = [
    "cabina", "salon", "vestibulo", "wc_normal", "bufet",
    "fuelles", "anexo", "bicicletas", "personal", "wc_pmr", "corredor"
]

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

HEIGHTS = {
    "seated": [0.10, 0.60, 1.00, 1.10],
    "standing": [0.10, 1.10, 1.70],
    "rh_co2": 1.10,
    "floor": 0.0
}

MARKER_COLORS = {
    "AT": "orange",
    "AT_standing": "red",
    "RH": "cyan",
    "CO2": "magenta",
    "TS_Fl": "green"
}

# ----------------------
# UTIL
# ----------------------
def px_to_m(pixels, mm_per_px):
    mm = pixels * mm_per_px
    return mm / 1000.0

def m_to_px(meters, mm_per_px):
    mm = meters * 1000.0
    return int(round(mm / mm_per_px))

def ensure_tesseract():
    if pytesseract is None:
        raise RuntimeError("pytesseract no está instalado. pip install pytesseract")
    try:
        _ = pytesseract.get_tesseract_version()
    except Exception:
        raise RuntimeError("Tesseract OCR no instalado en sistema. En Ubuntu: sudo apt install tesseract-ocr")

# ----------------------
# DETECCIÓN LÍNEA + OCR (igual que antes)
# ----------------------
def detect_scale_line_and_number(img):
    H, W = img.shape[:2]
    top_h = int(H * TOP_CROP_RATIO)
    top_region = img[0:top_h, :].copy()
    hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_MIN, HSV_MAX)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    min_len = int(W * 0.30)
    lines = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, minLineLength=min_len, maxLineGap=50)
    if lines is None:
        return None, None
    best_line = None
    best_len = 0
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = math.hypot(x2 - x1, y2 - y1)
        angle_deg = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle_deg < 10 and length > best_len:
            best_len = length
            best_line = (x1, y1, x2, y2)
    if best_line is None:
        return None, None
    x1, y1, x2, y2 = best_line
    line_len_px = math.hypot(x2 - x1, y2 - y1)
    # ROI encima de la línea (global coords)
    y_line_global = y2
    roi_y1 = max(0, y_line_global - int(top_h * 0.4) - 10)
    roi_y2 = max(0, y_line_global - 2)
    roi_x1 = max(0, int(min(x1, x2) - 0.05 * W))
    roi_x2 = min(W, int(max(x1, x2) + 0.05 * W))
    roi = img[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    if roi.size == 0:
        return line_len_px, None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        ensure_tesseract()
        cfg = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thr, config=cfg)
    except Exception:
        return line_len_px, None
    digits = re.findall(r'\d+', text)
    if not digits:
        return line_len_px, None
    number_str = max(digits, key=len)
    try:
        number_mm = int(number_str)
        return line_len_px, number_mm
    except:
        return line_len_px, None

# ----------------------
# MODO MANUAL (dos clicks)
# ----------------------
manual_points = []
def click_event_scale(event, x, y, flags, param):
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN:
        manual_points.append((x, y))
        print(f" → Punto marcado: {x}, {y}")

def get_scale_manual(img, filename):
    global manual_points
    manual_points = []
    clone = img.copy()
    cv2.namedWindow("Selecciona extremos de la escala")
    cv2.setMouseCallback("Selecciona extremos de la escala", click_event_scale)
    print("\n[MODO MANUAL] Marca dos extremos de la línea de escala y cierra la ventana.")
    while True:
        temp = clone.copy()
        for p in manual_points:
            cv2.circle(temp, p, 5, (0,0,255), -1)
        cv2.imshow("Selecciona extremos de la escala", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow("Selecciona extremos de la escala")
            return None, None
        if len(manual_points) == 2:
            break
    cv2.destroyWindow("Selecciona extremos de la escala")
    (x1,y1),(x2,y2) = manual_points
    line_len_px = math.hypot(x2 - x1, y2 - y1)
    print(f"Distancia en píxeles: {line_len_px:.2f}")
    while True:
        user_val = input(f"Introduce el número de la escala en mm para {filename}: ").strip()
        if user_val.lower() in ("skip","s"):
            return None, None
        try:
            number_mm = int(re.sub(r'\D', '', user_val))
            return line_len_px, number_mm
        except:
            print("Valor inválido. Escribe un entero (ej. 15000) o 'skip'.")

# ----------------------
# Subdivisión y sensores (cada bbox es una zona)
# ----------------------
def decide_subzones(L_m):
    if L_m < 5.0:
        return 1
    elif 5.0 <= L_m < 10.0:
        return 2
    else:
        return 3

def place_sensors_for_subzone_bbox(zone_name, bbox_px, sub_left_px, sub_right_px, sub_idx, total_sub, zone_global_idx, mm_per_px):
    """
    bbox_px: x1,y1,x2,y2,cls,conf (bbox original)
    sub_left_px, sub_right_px: px coords of subzone within bbox
    Returns sensors list (with x_m, height_m and name)
    """
    x1, y1, x2, y2, cls, conf = bbox_px
    Wb = sub_right_px - sub_left_px
    # coords in meters relative to bbox-leftmost global reference will be computed by caller
    left_m = px_to_m(sub_left_px, mm_per_px)
    right_m = px_to_m(sub_right_px, mm_per_px)
    width_m = right_m - left_m
    sensors = []
    # seated at 1/4 width
    quarter_px = sub_left_px + 0.25 * (sub_right_px - sub_left_px)
    # map x to meters in caller context
    # seated sensors (4 heights) at quarter x, heights per HEIGHTS['seated']
    for i, h in enumerate(HEIGHTS["seated"], start=1):
        name = f"S{zone_global_idx}.{sub_idx}_AT_{i:02d}"
        sensors.append({"name": name, "type": "AT", "zone": zone_name, "subzone": f"{zone_global_idx}.{sub_idx}",
                        "x_px": quarter_px, "height_m": h})
    # standing at center x
    center_px = sub_left_px + 0.5 * (sub_right_px - sub_left_px)
    for i, h in enumerate(HEIGHTS["standing"], start=1):
        name = f"S{zone_global_idx}.{sub_idx}_AT_S{i:02d}"
        sensors.append({"name": name, "type": "AT_standing", "zone": zone_name, "subzone": f"{zone_global_idx}.{sub_idx}",
                        "x_px": center_px, "height_m": h})
    # RH, CO2, floor (center)
    sensors.append({"name": f"S{zone_global_idx}.{sub_idx}_RH_01", "type": "RH", "zone": zone_name,
                    "subzone": f"{zone_global_idx}.{sub_idx}", "x_px": center_px, "height_m": HEIGHTS["rh_co2"]})
    sensors.append({"name": f"S{zone_global_idx}.{sub_idx}_CO2_01", "type": "CO2", "zone": zone_name,
                    "subzone": f"{zone_global_idx}.{sub_idx}", "x_px": center_px, "height_m": HEIGHTS["rh_co2"]})
    sensors.append({"name": f"S{zone_global_idx}.{sub_idx}_TS_FL_01", "type": "TS_Fl", "zone": zone_name,
                    "subzone": f"{zone_global_idx}.{sub_idx}", "x_px": center_px, "height_m": HEIGHTS["floor"]})
    # return sensors with x_px; caller will convert to x_m relative to global reference
    return sensors

# ----------------------
# Map sensor height (m) to pixel y inside bbox for drawing (heuristic)
# ----------------------
def map_height_to_bbox_y(h_m, bbox_y1, bbox_y2):
    """
    Heuristic mapping: 
    - floor (0.0) -> bottom (bbox_y2 - 3 px)
    - seated low (0.10) -> 0.7*height from top
    - seated higher (1.1) -> 0.45*height from top
    - standing heights -> 0.45 (1.1) and 0.25 (1.7) as relative positions
    This is approximate; heights are recorded exactly in Excel, drawing is visual aid.
    """
    h_rel = 0.5  # default center
    height = bbox_y2 - bbox_y1
    # choose heuristics
    if h_m == 0.0:
        y = bbox_y2 - 3
    elif h_m <= 0.60:
        y = int(bbox_y1 + 0.7 * height)
    elif h_m <= 1.1:
        y = int(bbox_y1 + 0.55 * height)
    elif h_m <= 1.7:
        y = int(bbox_y1 + 0.35 * height)
    else:
        y = int(bbox_y1 + 0.5 * height)
    return y

def draw_rect_alpha(img, x1,y1,x2,y2, color):
    overlay = img.copy()
    bgr = tuple(map(int, color[:3]))
    alpha = color[3]
    cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), bgr, 2)

# ----------------------
# Carga modelo
# ----------------------
print("Cargando modelo:", MODEL_PATH)
model = YOLO(MODEL_PATH)

# ----------------------
# Proceso principal
# ----------------------
image_files = sorted([p for p in Path(IMG_FOLDER).glob("*.*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]])
if not image_files:
    print("No hay imágenes en:", IMG_FOLDER); sys.exit(1)

for img_path_obj in image_files:
    img_path = str(img_path_obj)
    print("\n=== Procesando:", img_path_obj.name, "===")
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    # Escala: intentar automático
    line_len_px, number_mm = detect_scale_line_and_number(img)
    if line_len_px is None or number_mm is None:
        print(f"[AVISO] Detección automática fallida en {img_path_obj.name}. Activando modo manual.")
        line_len_px, number_mm = get_scale_manual(img, img_path_obj.name)
    if line_len_px is None or number_mm is None:
        print("[AVISO] No se obtuvo escala. Saltando imagen.")
        continue
    mm_per_px = number_mm / line_len_px
    print(f"Escala: {number_mm} mm en {line_len_px:.1f}px -> mm/px = {mm_per_px:.4f}")

    # Inferencia
    results = model.predict(source=img_path, conf=0.25, show=False)
    boxes_px = []
    for r in results:
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1,y1,x2,y2 = box.tolist()
            cls_id = int(cls_id); conf = float(conf)
            cls_name = model.names[cls_id] if cls_id in model.names else CLASS_NAMES[cls_id]
            boxes_px.append((x1,y1,x2,y2,cls_name,conf))

    if not boxes_px:
        print("No se detectaron bboxes. Siguiente imagen.")
        continue

    # calcular span global (para referencia si hace falta)
    global_xmin = min(b[0] for b in boxes_px)
    global_xmax = max(b[2] for b in boxes_px)
    span_px = global_xmax - global_xmin
    span_m = px_to_m(span_px, mm_per_px)
    print(f"Span total: {span_px:.1f}px -> {span_m:.3f} m")

    # Ahora: para CADA bbox (1B), generamos subzonas si es comfort zone
    sensors_list = []
    zone_counter = 1

    for bbox in boxes_px:
        x1,y1,x2,y2,cls,conf = bbox
        # bbox width in meters
        bbox_width_px = x2 - x1
        bbox_width_m = px_to_m(bbox_width_px, mm_per_px)
        # comfort zones subdivide
        if cls in ("salon","bufet"):
            n_sub = decide_subzones(bbox_width_m)
            # subzone px widths
            sub_w_px = bbox_width_px / n_sub if n_sub>0 else bbox_width_px
            for s in range(n_sub):
                sub_left_px = x1 + s*sub_w_px
                sub_right_px = sub_left_px + sub_w_px
                subsensors = place_sensors_for_subzone_bbox(cls, bbox, sub_left_px, sub_right_px, s+1, n_sub, zone_counter, mm_per_px)
                # convert x_px -> x_m relative to global_xmin reference (you wanted meters from leftmost bbox of image)
                for ss in subsensors:
                    ss["x_m"] = px_to_m(ss["x_px"] - global_xmin, mm_per_px)
                    ss["bbox"] = (x1,y1,x2,y2)
                sensors_list.extend(subsensors)
            zone_counter += 1
        elif cls == "cabina":
            # single sensor (seated) at bbox center if no seat detection
            cx_px = (x1 + x2)/2
            cx_m = px_to_m(cx_px - global_xmin, mm_per_px)
            sensors_list.append({"name": f"Cab{zone_counter}_AT_seated_01", "type":"AT", "zone":cls,
                                 "subzone":f"cab{zone_counter}.1", "x_m":cx_m, "height_m":HEIGHTS["seated"][0],
                                 "bbox":(x1,y1,x2,y2)})
            zone_counter += 1
        elif cls in ("wc_normal","wc_pmr"):
            cx_px = (x1 + x2)/2
            cx_m = px_to_m(cx_px - global_xmin, mm_per_px)
            sensors_list.append({"name": f"WC{zone_counter}_AT_01", "type":"AT", "zone":cls,
                                 "subzone":f"{cls}{zone_counter}.1", "x_m":cx_m, "height_m":1.1, "bbox":(x1,y1,x2,y2)})
            sensors_list.append({"name": f"WC{zone_counter}_RH_01", "type":"RH", "zone":cls,
                                 "subzone":f"{cls}{zone_counter}.1", "x_m":cx_m, "height_m":HEIGHTS["rh_co2"], "bbox":(x1,y1,x2,y2)})
            zone_counter += 1
        elif cls == "vestibulo":
            cx_px = (x1 + x2)/2; cx_m = px_to_m(cx_px - global_xmin, mm_per_px)
            sensors_list.append({"name": f"Vest{zone_counter}_AT_01", "type":"AT", "zone":cls,
                                 "subzone":f"{cls}{zone_counter}.1", "x_m":cx_m, "height_m":0.1, "bbox":(x1,y1,x2,y2)})
            sensors_list.append({"name": f"Vest{zone_counter}_AT_02", "type":"AT", "zone":cls,
                                 "subzone":f"{cls}{zone_counter}.1", "x_m":cx_m, "height_m":1.7, "bbox":(x1,y1,x2,y2)})
            zone_counter += 1
        else:
            # corredor, anexo, bicicletas, personal, fuelles -> standing sensors + RH
            cx_px = (x1 + x2)/2; cx_m = px_to_m(cx_px - global_xmin, mm_per_px)
            for i,h in enumerate(HEIGHTS["standing"], start=1):
                sensors_list.append({"name": f"{cls}{zone_counter}_AT_S{i:02d}", "type":"AT_standing", "zone":cls,
                                     "subzone":f"{cls}{zone_counter}.1", "x_m":cx_m, "height_m":h, "bbox":(x1,y1,x2,y2)})
            sensors_list.append({"name": f"{cls}{zone_counter}_RH_01", "type":"RH", "zone":cls,
                                 "subzone":f"{cls}{zone_counter}.1", "x_m":cx_m, "height_m":HEIGHTS["rh_co2"], "bbox":(x1,y1,x2,y2)})
            zone_counter += 1

    # Guardar Excel
    if sensors_list:
        # ensure x_m exists for all (cabina etc already have)
        for s in sensors_list:
            if "x_m" not in s and "x_px" in s:
                s["x_m"] = px_to_m(s["x_px"] - global_xmin, mm_per_px)
        df = pd.DataFrame(sensors_list)
        excel_path = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}_sensors.xlsx"
        df_out = df[["name","type","zone","subzone","x_m","height_m"]].copy()
        df_out.to_excel(excel_path, index=False)
        print("Excel guardado en:", excel_path)

    # Dibujar imagen y PDF: ahora respetamos bbox (no estiramos)
    img_vis = img.copy()
    # draw each bbox as filled rect (its bbox area)
    for (x1,y1,x2,y2,cls,conf) in boxes_px:
        color = CLASS_COLORS.get(cls, (120,120,120,0.25))
        draw_rect_alpha(img_vis, x1,y1,x2,y2, color)
        col = tuple(map(int, color[:3]))
        cv2.rectangle(img_vis, (int(x1),int(y1)), (int(x2),int(y2)), col, 2)
        cv2.putText(img_vis, f"{cls} {conf:.2f}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)

    # draw subzone vertical dashed lines inside bbox for comfort zones
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)); ax.axis('off')
    for bbox in boxes_px:
        x1,y1,x2,y2,cls,conf = bbox
        if cls in ("salon","bufet"):
            bbox_w_px = x2 - x1
            bbox_w_m = px_to_m(bbox_w_px, mm_per_px)
            n_sub = decide_subzones(bbox_w_m)
            if n_sub > 1:
                sub_w_px = bbox_w_px / n_sub
                color_rgb = np.array(CLASS_COLORS.get(cls,(120,120,120,0.25))[:3]) / 255.0
                for s in range(1, n_sub):
                    xline = x1 + s * sub_w_px
                    ax.add_line(Line2D([xline,xline],[y1,y2], linestyle=(0,(5,5)), color=color_rgb, linewidth=2))

    # draw sensors as stars mapped into bbox vertical positions using heuristic
    for s in sensors_list:
        # find bbox for this sensor
        bbox = s.get("bbox", None)
        if bbox:
            bx1,b_y1,bx2,b_y2 = bbox
            x_px = (s.get("x_px", None) if s.get("x_px", None) is not None else (global_xmin + m_to_px(s["x_m"], mm_per_px)))
            # if x_px is global (not relative to global_xmin), adjust
            if x_px is not None and isinstance(x_px, (int,float)):
                # map to axes (matplotlib uses pixel coords)
                y_px = map_height_to_bbox_y(s["height_m"], int(b_y1), int(b_y2))
                ax.plot(x_px, y_px, marker='*', markersize=10, color=MARKER_COLORS.get(s["type"].split('_')[0],"white"))
                ax.text(x_px+3, y_px+3, s["name"], color='white', fontsize=7)
        else:
            # safety: place roughly on midline
            x_px = global_xmin + m_to_px(s["x_m"], mm_per_px)
            y_px = int(H*0.5)
            ax.plot(x_px, y_px, marker='*', markersize=10, color=MARKER_COLORS.get(s["type"].split('_')[0],"white"))
            ax.text(x_px+3, y_px+3, s["name"], color='white', fontsize=7)

    legend_handles = [Line2D([0],[0], marker='*', color='w', label=k, markerfacecolor=v, markersize=10) for k,v in MARKER_COLORS.items()]
    ax.legend(handles=legend_handles, loc='lower right', fontsize='small')

    pdf_path = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}_annotated.pdf"
    img_out_path = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}_annotated.jpg"
    fig.savefig(str(pdf_path), bbox_inches='tight', dpi=200)
    plt.close(fig)
    cv2.imwrite(str(img_out_path), img_vis)

    # Guardar detecciones .txt (YOLO normalized)
    dets_txt = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}.txt"
    with open(dets_txt, "w") as f:
        for (x1,y1,x2,y2,cls,conf) in boxes_px:
            xc = ((x1+x2)/2)/W
            yc = ((y1+y2)/2)/H
            bw = (x2-x1)/W
            bh = (y2-y1)/H
            cls_id = CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 0
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Guardado: PDF={pdf_path}, IMG={img_out_path}, EXCEL={excel_path}, DETS={dets_txt}")

print("\nProceso terminado. Resultados en:", OUTPUT_FOLDER)
