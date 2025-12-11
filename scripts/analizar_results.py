"""
scripts/evaluate_yolov8_visual.py

Evalúa un modelo YOLOv8 entrenado sobre un conjunto de validación:
- Calcula IoU y RMSE por imagen.
- Genera gráfica de RMSE por imagen.
- Crea imágenes con bboxes predichas vs ground truth.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# Rutas a ajustar
# -----------------------------
MODEL_PATH = "/home/pablo/Documents/pro_vision/planos/models/yolo_trenes_better_bbox/weights/best.pt"
VAL_DIR =           "/home/pablo/Documents/pro_vision/planos/val/images"
VAL_LABELS_DIR =    "/home/pablo/Documents/pro_vision/planos/val/labels"
OUTPUT_DIR = "/home/pablo/Documents/pro_vision/planos/val/output_results"  # donde guardar imágenes de salida
IOU_THRESHOLD = 0.2  # para considerar acierto

def load_yolo_labels(txt_file, w, h):
    boxes = []
    txt_path = Path(txt_file)
    if not txt_path.exists():
        return boxes
    with open(txt_path, "r") as f:
        for line in f.readlines():
            cls, xc, yc, bw, bh = map(float, line.split())
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            boxes.append([int(cls), x1, y1, x2, y2])
    return boxes

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0.0

def compute_iou_rmse(pred_boxes, gt_boxes):
    ious = []
    for pb, gb in zip(pred_boxes, gt_boxes):
        ious.append(iou(pb[1:], gb[1:]))
    ious = np.array(ious)
    rmse = np.sqrt(np.mean((1-ious)**2))
    return rmse, ious

def draw_boxes_translucid(img, pred_boxes, gt_boxes, ious, iou_threshold=IOU_THRESHOLD):
    out = img.copy()
    overlay = out.copy()

    for idx, (pb, gb) in enumerate(zip(pred_boxes, gt_boxes)):
        # Color según acierto/fallo
        color_fill = (0,255,0) if ious[idx] >= iou_threshold else (0,0,255)
        color_border = (0,200,0) if ious[idx] >= iou_threshold else (0,0,200)

        # Área de intersección
        x1 = max(pb[1], gb[1])
        y1 = max(pb[2], gb[2])
        x2 = min(pb[3], gb[3])
        y2 = min(pb[4], gb[4])

        # Dibujar bbox predicha y GT borde más saturado
        cv2.rectangle(out, (pb[1], pb[2]), (pb[3], pb[4]), color_border, 2)
        cv2.rectangle(out, (gb[1], gb[2]), (gb[3], gb[4]), (255,0,0), 1)

        # Dibujar relleno semitransparente en la intersección
        if x2 > x1 and y2 > y1:
            alpha = 0.4
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_fill, -1)
            cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    return out

def evaluate_model_visual(model_path, val_dir, val_labels_dir, output_dir):
    from ultralytics import YOLO
    model = YOLO(model_path)
    val_images = sorted(Path(val_dir).glob("*.jpg"))
    all_rmse = []
    image_names = []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in val_images:
        img_cv = cv2.imread(str(img_path))
        h, w = img_cv.shape[:2]

        results = model.predict(str(img_path), imgsz=640, conf=0.25)
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                pred_boxes.append([cls, int(x1), int(y1), int(x2), int(y2)])

        gt_path = Path(val_labels_dir) / (img_path.stem + ".txt")
        gt_boxes = load_yolo_labels(gt_path, w, h)

        if pred_boxes and gt_boxes:
            rmse, ious = compute_iou_rmse(pred_boxes, gt_boxes)
            all_rmse.append(rmse)
            image_names.append(img_path.name)
            print(f"{img_path.name} - RMSE IoU: {rmse:.4f}")

            out_img = draw_boxes_translucid(img_cv, pred_boxes, gt_boxes, ious)
            out_path = output_dir / f"{img_path.stem}_compare.jpg"
            cv2.imwrite(str(out_path), out_img)
        else:
            print(f"{img_path.name} - Sin bboxes para comparar")

    # Gráfica RMSE
    if all_rmse:
        total_rmse = np.mean(all_rmse)
        print(f"\nRMSE IoU total de validación: {total_rmse:.4f}")

        plt.figure(figsize=(10,5))
        plt.bar(image_names, all_rmse, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("RMSE IoU")
        plt.title("RMSE IoU por imagen")
        plt.tight_layout()
        plt.savefig(output_dir / "rmse_iou_per_image.png")
        plt.show()
    else:
        print("\nNo se pudieron calcular RMSE IoU para ninguna imagen.")

if __name__ == "__main__":
    evaluate_model_visual(MODEL_PATH, VAL_DIR, VAL_LABELS_DIR, OUTPUT_DIR)