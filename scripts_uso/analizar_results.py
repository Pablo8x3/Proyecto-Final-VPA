#!/usr/bin/env python3
"""
analysis_yolov8_internal.py

Script autónomo para analizar un modelo YOLO ya entrenado usando IoU.
No usa parámetros por terminal: TODAS LAS RUTAS SE DEFINEN AQUÍ DENTRO.

Qué hace:
  - Carga best.pt o cualquier modelo YOLOv8
  - Evalúa IoU sobre train / val / test
  - Genera por split:
        * iou_per_image.csv
        * iou_per_class.csv
        * iou_summary.json
        * iou_summary.txt
  - Guarda todo en una carpeta de resultados

Requisitos:
    pip install ultralytics numpy pandas opencv-python pyyaml
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------
# ---------------- CONFIGURACIÓN DEL USUARIO --------------
# ---------------------------------------------------------

# Modelo YOLO a analizar (.pt)
MODEL_PT = Path("planos/models/yolo_trenes/weights/best.pt")

# Directorios split (o los que quieras analizar)
TRAIN_IMAGES = Path("planos/train/images")
VAL_IMAGES   = Path("planos/val/images")
TEST_IMAGES  = Path("planos/comprobar_manual/images/")

TRAIN_LABELS = Path("planos/train/labels")
VAL_LABELS   = Path("planos/val/labels/")
TEST_LABELS  = Path("planos/comprobar_manual/labels/")

# YAML con la lista de clases
NAMES_YAML = Path("planos/data_trenes.yaml")

# Carpeta donde se guardarán los informes de análisis
OUTPUT_DIR = Path("planos/models/yolo_trenes/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# ---------------- FUNCIONES AUXILIARES -------------------
# ---------------------------------------------------------

def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".png"}
    return sorted([p for p in Path(folder).glob("*") if p.suffix.lower() in exts])


def normalize_box_coords(x1, y1, x2, y2):
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return int(x1), int(y1), int(x2), int(y2)


def read_yolo_txt_as_boxes(txt_path, img_w, img_h):
    """ Devuelve lista de [x1,y1,x2,y2,cls] """
    boxes = []
    txt_path = Path(txt_path)

    if not txt_path.exists():
        return boxes

    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            cls = int(float(p[0]))
            xc, yc, bw, bh = map(float, p[1:])
            x1 = (xc - bw/2.0) * img_w
            x2 = (xc + bw/2.0) * img_w
            y1 = (yc - bh/2.0) * img_h
            y2 = (yc + bh/2.0) * img_h
            x1, y1, x2, y2 = normalize_box_coords(x1, y1, x2, y2)
            boxes.append([x1, y1, x2, y2, cls])
    return boxes


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    areaB = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))

    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return interArea / union

# ---------------------------------------------------------
# ---------------------- EVALUACIÓN -----------------------
# ---------------------------------------------------------

def evaluate_model(model_path, images, labels_dir, names_list):
    from ultralytics import YOLO

    print(f"\n[INFO] Cargando modelo {model_path}")
    model = YOLO(str(model_path))

    per_image_rows = []
    per_class_ious = {i: [] for i in range(len(names_list))}
    all_ious = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        gt_boxes = read_yolo_txt_as_boxes(labels_dir / f"{img_path.stem}.txt", W, H)

        results = model.predict(str(img_path), conf=0.001, iou=0.5, verbose=False)

        preds = []
        for r in results:
            if hasattr(r, "boxes") and r.boxes is not None:
                for b, c in zip(r.boxes.xyxy, r.boxes.cls):
                    x1, y1, x2, y2 = map(float, b.tolist())
                    preds.append([int(x1), int(y1), int(x2), int(y2), int(c)])

        # IoU por imagen
        ious_img = []
        for gt in gt_boxes:
            gt_box = gt[:4]
            best = 0.0
            for pr in preds:
                best = max(best, compute_iou(gt_box, pr[:4]))
            ious_img.append(best)
            all_ious.append(best)
            per_class_ious[gt[4]].append(best)

        per_image_rows.append({
            "image": img_path.name,
            "n_gt": len(gt_boxes),
            "n_pred": len(preds),
            "mean_iou": float(np.mean(ious_img)) if ious_img else 0.0,
            "min_iou": float(np.min(ious_img)) if ious_img else 0.0,
            "max_iou": float(np.max(ious_img)) if ious_img else 0.0
        })

    # Resumen por clase
    per_class_summary = {}
    for cls_id, vals in per_class_ious.items():
        per_class_summary[cls_id] = {
            "class_name": names_list[cls_id],
            "mean_iou": float(np.mean(vals)) if vals else None,
            "std_iou": float(np.std(vals)) if vals else None,
            "count": len(vals)
        }

    summary = {
        "global_mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "global_std_iou": float(np.std(all_ious)) if all_ious else 0.0,
        "total_gt_boxes": len(all_ious),
        "per_class": per_class_summary
    }

    return summary, per_image_rows


# ---------------------------------------------------------
# ------------------------ MAIN ---------------------------
# ---------------------------------------------------------

def main():

    # cargar nombres
    with open(NAMES_YAML, "r") as f:
        names_list = yaml.safe_load(f)["names"]

    splits = {
        "train": (list_images(TRAIN_IMAGES), TRAIN_LABELS),
        "val":   (list_images(VAL_IMAGES),   VAL_LABELS),
        "test":  (list_images(TEST_IMAGES),  TEST_LABELS)
    }

    for split_name, (imgs, lbl_dir) in splits.items():
        if not imgs:
            print(f"[WARN] No hay imágenes en {split_name}, se omite.")
            continue

        print(f"\n[INFO] Evaluando {split_name} ({len(imgs)} imágenes)...")

        summary, per_image = evaluate_model(MODEL_PT, imgs, lbl_dir, names_list)

        # CSV de imágenes
        df = pd.DataFrame(per_image)
        df.to_csv(OUTPUT_DIR / f"{split_name}_iou_per_image.csv", index=False)

        # CSV por clase
        rows = []
        for cls_id, info in summary["per_class"].items():
            rows.append({
                "class_id": cls_id,
                "class_name": info["class_name"],
                "mean_iou": info["mean_iou"],
                "std_iou": info["std_iou"],
                "count": info["count"]
            })
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"{split_name}_iou_per_class.csv", index=False)

        # JSON
        with open(OUTPUT_DIR / f"{split_name}_iou_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # TXT
        with open(OUTPUT_DIR / f"{split_name}_iou_summary.txt", "w") as f:
            f.write(f"Split: {split_name}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n\n")
            f.write(f"Images: {len(imgs)}\n")
            f.write(f"Global mean IoU: {summary['global_mean_iou']:.5f}\n")
            f.write(f"Std: {summary['global_std_iou']:.5f}\n")
            f.write(f"Total GT boxes: {summary['total_gt_boxes']}\n\n")
            f.write("Per-class IoU:\n")
            for cls_id, info in summary["per_class"].items():
                f.write(f"  - {info['class_name']} (id {cls_id}): mean={info['mean_iou']} std={info['std_iou']} count={info['count']}\n")

    print("\n[OK] Análisis completado. Resultados en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
