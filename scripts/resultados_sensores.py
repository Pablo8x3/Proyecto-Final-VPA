#!/usr/bin/env python3
# scripts/test_yolov8_full_final.py
# Integrado: detección escala automática (línea + OCR) con fallback manual (dos clicks),
# inferencia YOLOv8, cada bbox = zona, subdivisión de comfort zones en 1/2/3 subzonas,
# sensores colocados geométricamente (seated: two sets on diagonal @ 0.25 & 0.75,
# standing/RH/CO2/floor: centroid of subzone).
#
# Requisitos: pip install ultralytics opencv-python numpy pandas matplotlib openpyxl pytesseract
# Además: tesseract OCR en sistema (ej. Ubuntu: sudo apt install tesseract-ocr)
#
# AJUSTA las rutas al inicio si hace falta.

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
MODEL_PATH = "/home/pablo/Documents/pro_vision/planos/models/yolo_trenes_better_bbox/weights/best.pt"
IMG_FOLDER = "/home/pablo/Documents/pro_vision/planos/comprobar_manual_better_bbox/images"
OUTPUT_FOLDER = "/home/pablo/Documents/pro_vision/planos/comprobar_manual_better_bbox/results_final"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TOP_CROP_RATIO = 0.20  # zona superior para buscar la línea
HSV_MIN = np.array([35, 20, 40])
HSV_MAX = np.array([95, 200, 255])

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50

# gap for merging (kept for potential future use)
MERGE_GAP_M = 0.10  # 10 cm

CLASS_NAMES = [
    "cabina", "salon", "vestibulo", "wc_normal", "bufet",
    "fuelles", "anexo", "bicicletas", "personal", "wc_pmr", "corredor"
]

# Colors (B,G,R,alpha)
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
        raise RuntimeError("Tesseract OCR no está instalado en el sistema. En Ubuntu: sudo apt install tesseract-ocr")

# ----------------------
# DETECCIÓN LÍNEA + OCR (usando tu implementación mejorada)
# ----------------------
def detect_scale_line_and_number(img):
    H, W = img.shape[:2]
    top_h = int(H * TOP_CROP_RATIO)
    top_region = img[0:top_h, :].copy()
    gray_top = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)

    # edge detection (your version works well)
    edges = cv2.Canny(gray_top, 50, 150, apertureSize=3)
    minLineLen = max(50, int(W * 0.25))
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=minLineLen, maxLineGap=50)

    if lines is None:
        return None, None

    # choose longest near-horizontal
    best_line = None
    max_length = 0
    for l in lines:
        x1,y1,x2,y2 = l[0]
        length = math.hypot(x2-x1, y2-y1)
        angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
        if (angle < 10 or angle > 170) and length > max_length:
            max_length = length
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        return None, None

    x1,y1,x2,y2 = best_line
    line_len_px = math.hypot(x2-x1, y2-y1)

    # ROI above the line (global coords): map from top_region to full image
    # note: top_region starts at y=0 so coords line up
    y_line_global = int((y1 + y2) / 2)
    roi_y1 = max(0, y_line_global - int(top_h * 0.5))
    roi_y2 = max(0, y_line_global + 10)
    roi_x1 = max(0, int(min(x1, x2) - 0.05 * W))
    roi_x2 = min(W, int(max(x1, x2) + 0.05 * W))
    roi = top_region[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    if roi.size == 0:
        return line_len_px, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        ensure_tesseract()
        cfg = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
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
    print("\n[MODO MANUAL] Marca dos extremos de la línea de escala y cierra la ventana (o presiona ESC para cancelar).")
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
# GEOM HELPERS
# ----------------------
def get_center_of_subzone(sub_left_px, sub_right_px, y1, y2):
    cx = (sub_left_px + sub_right_px) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy

def get_diagonal_point(sub_left_px, sub_right_px, y1, y2, t):
    # diagonal bottom-left -> top-right
    x_start, y_start = sub_left_px, y2
    x_end, y_end = sub_right_px, y1
    x = x_start + t * (x_end - x_start)
    y = y_start + t * (y_end - y_start)
    return x, y

# ----------------------
# DRAW HELPERS
# ----------------------
def draw_rect_alpha(img, x1, y1, x2, y2, color):
    overlay = img.copy()
    bgr = tuple(map(int, color[:3]))
    alpha = float(color[3])
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 2)

# ----------------------
# CARGA MODELO
# ----------------------
print("Cargando modelo:", MODEL_PATH)
model = YOLO(MODEL_PATH)

# ----------------------
# BUCLE PRINCIPAL
# ----------------------
image_files = sorted([p for p in Path(IMG_FOLDER).glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
if not image_files:
    print("No hay imágenes en:", IMG_FOLDER)
    sys.exit(1)

for img_path_obj in image_files:
    img_path = str(img_path_obj)
    print("\n=== Procesando:", img_path_obj.name, "===")
    img = cv2.imread(img_path)
    if img is None:
        print("No se pudo leer la imagen:", img_path)
        continue
    H, W = img.shape[:2]

    # 1) Escala: intentar automática (tu detector mejorado), con fallback manual
    # First try automatic using detect_scale_line_and_number (Hough + OCR)
    line_len_px, number_mm = detect_scale_line_and_number(img)
    if line_len_px is None or number_mm is None:
        # Try a second, slightly different automatic approach based on edges + Hough (robustness)
        try:
            # Convert to gray and try full-image Hough with different parameters
            gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges_full = cv2.Canny(gray_full, 50, 150, apertureSize=3)
            lines_full = cv2.HoughLinesP(edges_full, 1, np.pi/180, threshold=120, minLineLength=max(100, int(W*0.2)), maxLineGap=30)
            best_line2 = None
            max_len2 = 0
            if lines_full is not None:
                for l in lines_full:
                    x1,y1,x2,y2 = l[0]
                    length = math.hypot(x2-x1, y2-y1)
                    angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
                    if (angle < 12 or angle > 168) and length > max_len2:
                        max_len2 = length
                        best_line2 = (x1,y1,x2,y2)
            if best_line2 is not None:
                # attempt OCR using a ROI around that line
                bx1,by1,bx2,by2 = best_line2  # intentionally will fail if malformed; except below will catch
                # map ROI to top region heuristics then OCR similarly (reuse detect_scale_line_and_number logic)
                line_len_px_alt = math.hypot(bx2-bx1, by2-by1)
                # attempt to read OCR over top area again
                # reuse function but if fails we'll go to manual
                # (call detect_scale_line_and_number which uses top_region approach)
                line_len_px_tmp, number_mm_tmp = detect_scale_line_and_number(img)
                if line_len_px_tmp is not None and number_mm_tmp is not None:
                    line_len_px, number_mm = line_len_px_tmp, number_mm_tmp
        except Exception:
            pass

    if line_len_px is None or number_mm is None:
        print(f"[AVISO] Detección automática fallida en {img_path_obj.name}. Activando modo manual.")
        line_len_px, number_mm = get_scale_manual(img, img_path_obj.name)

    if line_len_px is None or number_mm is None:
        print("[AVISO] No se obtuvo escala. Saltando imagen.")
        continue

    mm_per_px = number_mm / line_len_px
    print(f"Escala: {number_mm} mm en {line_len_px:.1f}px -> mm/px = {mm_per_px:.6f}")

    # 2) Inferencia YOLO
    results = model.predict(source=img_path, conf=0.25, show=False)
    boxes_px = []
    for r in results:
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            cls_id = int(cls_id); conf = float(conf)
            cls_name = model.names[cls_id] if cls_id in model.names else CLASS_NAMES[cls_id]
            boxes_px.append((x1, y1, x2, y2, cls_name, conf))

    if not boxes_px:
        print("No se detectaron bboxes. Siguiente imagen.")
        continue

    # global refs (for meters origin)
    global_xmin = min(b[0] for b in boxes_px)
    global_ymin = min(b[1] for b in boxes_px)
    global_xmax = max(b[2] for b in boxes_px)
    global_ymax = max(b[3] for b in boxes_px)
    span_px = global_xmax - global_xmin
    span_m = px_to_m(span_px, mm_per_px)
    print(f"Span total: {span_px:.1f}px -> {span_m:.3f} m")

    # 3) Generar sensores (cada bbox = zona)
    sensors_list = []
    zone_counter = 1

    for bbox in boxes_px:
        x1, y1, x2, y2, cls, conf = bbox
        bbox_w_px = x2 - x1
        bbox_w_m = px_to_m(bbox_w_px, mm_per_px)

        if cls in ("salon", "bufet"):
            n_sub = 1
            if bbox_w_m < 5.0:
                n_sub = 1
            elif 5.0 <= bbox_w_m < 10.0:
                n_sub = 2
            else:
                n_sub = 3
            sub_w_px = bbox_w_px / n_sub if n_sub > 0 else bbox_w_px
            for s in range(n_sub):
                sub_left_px = x1 + s * sub_w_px
                sub_right_px = sub_left_px + sub_w_px
                # centroid of subzone (geom center)
                cx_px, cy_px = get_center_of_subzone(sub_left_px, sub_right_px, y1, y2)

                # SEATED: two sets on diagonal at t=0.25 and t=0.75, each set has 4 heights
                for t in (0.25, 0.75):
                    px_pt_x, px_pt_y = get_diagonal_point(sub_left_px, sub_right_px, y1, y2, t)
                    for i, h in enumerate(HEIGHTS["seated"], start=1):
                        name = f"S{zone_counter}.{s+1}_AT_{i:02d}"
                        x_m = px_to_m(px_pt_x - global_xmin, mm_per_px)
                        y_m = px_to_m(px_pt_y - global_ymin, mm_per_px)
                        sensors_list.append({
                            "name": name, "type": "AT", "zone": cls, "subzone": f"{zone_counter}.{s+1}",
                            "x_px": px_pt_x, "y_px": px_pt_y, "x_m": x_m, "y_m": y_m, "height_m": h,
                            "bbox": (x1, y1, x2, y2)
                        })
                # STANDING: center geom of subzone -> 3 heights
                for i, h in enumerate(HEIGHTS["standing"], start=1):
                    name = f"S{zone_counter}.{s+1}_AT_S{i:02d}"
                    x_m = px_to_m(cx_px - global_xmin, mm_per_px)
                    y_m = px_to_m(cy_px - global_ymin, mm_per_px)
                    sensors_list.append({
                        "name": name, "type": "AT_standing", "zone": cls, "subzone": f"{zone_counter}.{s+1}",
                        "x_px": cx_px, "y_px": cy_px, "x_m": x_m, "y_m": y_m, "height_m": h,
                        "bbox": (x1, y1, x2, y2)
                    })
                # RH, CO2, Floor -> center geom of subzone
                sensors_list.append({
                    "name": f"S{zone_counter}.{s+1}_RH_01", "type": "RH", "zone": cls, "subzone": f"{zone_counter}.{s+1}",
                    "x_px": cx_px, "y_px": cy_px, "x_m": px_to_m(cx_px - global_xmin, mm_per_px),
                    "y_m": px_to_m(cy_px - global_ymin, mm_per_px), "height_m": HEIGHTS["rh_co2"],
                    "bbox": (x1, y1, x2, y2)
                })
                sensors_list.append({
                    "name": f"S{zone_counter}.{s+1}_CO2_01", "type": "CO2", "zone": cls, "subzone": f"{zone_counter}.{s+1}",
                    "x_px": cx_px, "y_px": cy_px, "x_m": px_to_m(cx_px - global_xmin, mm_per_px),
                    "y_m": px_to_m(cy_px - global_ymin, mm_per_px), "height_m": HEIGHTS["rh_co2"],
                    "bbox": (x1, y1, x2, y2)
                })
                sensors_list.append({
                    "name": f"S{zone_counter}.{s+1}_TS_FL_01", "type": "TS_Fl", "zone": cls, "subzone": f"{zone_counter}.{s+1}",
                    "x_px": cx_px, "y_px": cy_px, "x_m": px_to_m(cx_px - global_xmin, mm_per_px),
                    "y_m": px_to_m(cy_px - global_ymin, mm_per_px), "height_m": HEIGHTS["floor"],
                    "bbox": (x1, y1, x2, y2)
                })
            zone_counter += 1

        elif cls == "cabina":
            cx_px = (x1 + x2) / 2.0
            cy_px = (y1 + y2) / 2.0
            name = f"Cab{zone_counter}_AT_seated_01"
            sensors_list.append({
                "name": name, "type": "AT", "zone": "cabina", "subzone": f"cab{zone_counter}.1",
                "x_px": cx_px, "y_px": cy_px, "x_m": px_to_m(cx_px - global_xmin, mm_per_px),
                "y_m": px_to_m(cy_px - global_ymin, mm_per_px), "height_m": HEIGHTS["seated"][0],
                "bbox": (x1, y1, x2, y2)
            })
            zone_counter += 1

        elif cls in ("wc_normal", "wc_pmr"):
            cx_px = (x1 + x2) / 2.0
            cy_px = (y1 + y2) / 2.0
            sensors_list.append({
                "name": f"WC{zone_counter}_AT_01", "type": "AT", "zone": cls, "subzone": f"{cls}{zone_counter}.1",
                "x_px": cx_px, "y_px": cy_px, "x_m": px_to_m(cx_px - global_xmin, mm_per_px),
                "y_m": px_to_m(cy_px - global_ymin, mm_per_px), "height_m": 1.1, "bbox": (x1, y1, x2, y2)
            })
            sensors_list.append({
                "name": f"WC{zone_counter}_RH_01", "type": "RH", "zone": cls, "subzone": f"{cls}{zone_counter}.1",
                "x_px": cx_px, "y_px": cy_px, "x_m": px_to_m(cx_px - global_xmin, mm_per_px),
                "y_m": px_to_m(cy_px - global_ymin, mm_per_px), "height_m": HEIGHTS["rh_co2"], "bbox": (x1, y1, x2, y2)
            })
            zone_counter += 1

        elif cls == "vestibulo":
            cx_px = (x1 + x2) / 2.0
            cy_px = (y1 + y2) / 2.0
            sensors_list.append({
                "name": f"Vest{zone_counter}_AT_01", "type": "AT", "zone": cls, "subzone": f"{cls}{zone_counter}.1",
                "x_px": cx_px, "y_px": cy_px, "x_m": px_to_m(cx_px - global_xmin, mm_per_px),
                "y_m": px_to_m(cy_px - global_ymin, mm_per_px), "height_m": 0.1, "bbox": (x1, y1, x2, y2)
            })
            sensors_list.append({
                "name": f"Vest{zone_counter}_AT_02", "type": "AT", "zone": cls, "subzone": f"{cls}{zone_counter}.1",
                "x_px": cx_px, "y_px": cy_px, "x_m": px_to_m(cx_px - global_xmin, mm_per_px),
                "y_m": px_to_m(cy_px - global_ymin, mm_per_px), "height_m": 1.7, "bbox": (x1, y1, x2, y2)
            })
            zone_counter += 1

        else:
            # corredor, anexo, bicicletas, personal, fuelles -> standing + RH center
            cx_px = (x1 + x2) / 2.0
            cy_px = (y1 + y2) / 2.0
            for i, h in enumerate(HEIGHTS["standing"], start=1):
                sensors_list.append({
                    "name": f"{cls}{zone_counter}_AT_S{i:02d}", "type": "AT_standing", "zone": cls, "subzone": f"{cls}{zone_counter}.1",
                    "x_px": cx_px, "y_px": cy_px, "x_m": px_to_m(cx_px - global_xmin, mm_per_px),
                    "y_m": px_to_m(cy_px - global_ymin, mm_per_px), "height_m": h, "bbox": (x1, y1, x2, y2)
                })
            sensors_list.append({
                "name": f"{cls}{zone_counter}_RH_01", "type": "RH", "zone": cls, "subzone": f"{cls}{zone_counter}.1",
                "x_px": cx_px, "y_px": cy_px, "x_m": px_to_m(cx_px - global_xmin, mm_per_px),
                "y_m": px_to_m(cy_px - global_ymin, mm_per_px), "height_m": HEIGHTS["rh_co2"], "bbox": (x1, y1, x2, y2)
            })
            zone_counter += 1

    # 4) Guardar Excel (x_m, y_m, height_m)
    if sensors_list:
        df = pd.DataFrame(sensors_list)
        excel_path = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}_sensors.xlsx"
        # Ensure columns exist
        out_cols = ["name", "type", "zone", "subzone", "x_m", "y_m", "height_m"]
        # If some sensors lack y_m (unlikely), compute fallback from y_px
        for s in sensors_list:
            if "y_m" not in s and "y_px" in s:
                s["y_m"] = px_to_m(s["y_px"] - global_ymin, mm_per_px)
        df_out = pd.DataFrame([{k: s.get(k, None) for k in out_cols} for s in sensors_list])
        df_out.to_excel(excel_path, index=False)
        print("Excel guardado en:", excel_path)

    # 5) Dibujo: respetar bboxes y marcar sensors on-plane
    img_vis = img.copy()
    # draw bboxes filled
    for (bx1, by1, bx2, by2, bcls, bconf) in boxes_px:
        color = CLASS_COLORS.get(bcls, (120,120,120,0.25))
        draw_rect_alpha(img_vis, bx1, by1, bx2, by2, color)
        col = tuple(map(int, color[:3]))
        cv2.rectangle(img_vis, (int(bx1), int(by1)), (int(bx2), int(by2)), col, 2)
        cv2.putText(img_vis, f"{bcls} {bconf:.2f}", (int(bx1), int(by1)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)

    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    # draw subzone separators (dashed) inside bboxes for comfort
    for (bx1, by1, bx2, by2, bcls, bconf) in boxes_px:
        if bcls in ("salon", "bufet"):
            bbox_w_px = bx2 - bx1
            bbox_w_m = px_to_m(bbox_w_px, mm_per_px)
            n_sub = 1
            if bbox_w_m < 5.0:
                n_sub = 1
            elif 5.0 <= bbox_w_m < 10.0:
                n_sub = 2
            else:
                n_sub = 3
            if n_sub > 1:
                sub_w_px = bbox_w_px / n_sub
                color_rgb = np.array(CLASS_COLORS.get(bcls, (120,120,120,0.25))[:3]) / 255.0
                for s in range(1, n_sub):
                    xline = bx1 + s * sub_w_px
                    ax.add_line(Line2D([xline, xline], [by1, by2], linestyle=(0, (5,5)), color=color_rgb, linewidth=2))

    # plot sensors at (x_px, y_px)
    for s in sensors_list:
        x_px = s.get("x_px", None)
        y_px = s.get("y_px", None)
        if x_px is None or y_px is None:
            # fallback compute from x_m,y_m
            x_px = global_xmin + (s["x_m"] * 1000.0) / mm_per_px
            y_px = global_ymin + (s["y_m"] * 1000.0) / mm_per_px
        color = MARKER_COLORS.get(s["type"].split('_')[0], "white")
        ax.plot(x_px, y_px, marker='*', markersize=10, color=color)
        ax.text(x_px + 3, y_px + 3, s["name"], color='white', fontsize=7)

    legend_handles = [Line2D([0],[0], marker='*', color='w', label=k, markerfacecolor=v, markersize=10)
                      for k, v in MARKER_COLORS.items()]
    ax.legend(handles=legend_handles, loc='lower right', fontsize='small')

    pdf_path = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}_annotated.pdf"
    img_out_path = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}_annotated.jpg"
    fig.savefig(str(pdf_path), bbox_inches='tight', dpi=200)
    plt.close(fig)
    cv2.imwrite(str(img_out_path), img_vis)

    # 6) Guardar detecciones .txt (YOLO normalized)
    dets_txt = Path(OUTPUT_FOLDER) / f"{img_path_obj.stem}.txt"
    with open(dets_txt, "w") as f:
        for (x1, y1, x2, y2, cls, conf) in boxes_px:
            xc = ((x1 + x2) / 2) / W
            yc = ((y1 + y2) / 2) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            cls_id = CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 0
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Guardado: PDF={pdf_path}, IMG={img_out_path}, EXCEL={excel_path}, DETS={dets_txt}")

print("\nProceso terminado. Resultados en:", OUTPUT_FOLDER)
