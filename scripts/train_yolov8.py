#!/usr/bin/env python3
"""
scripts/train_yolov8_with_split_and_iou.py

Funciones añadidas al flujo de entrenamiento YOLOv8:
  - División automática 70/20/10 (train/val/test) desde una carpeta dataset/images + dataset/labels
    (se crean symlinks en planos/split/...). Split reproducible usando semilla.
  - Contador de entrenamientos: crea carpetas de salida nombradas entrenamiento_{N}
    incrementando N automáticamente (archivo train_counter.txt).
  - Entrena YOLOv8 sobre el split train/val y guarda resultados en la carpeta de ese entrenamiento.
  - Al finalizar carga best.pt y evalúa el modelo en el split test usando IoU entre GT y pred.
  - Guarda métricas (por imagen, por clase y globales) en CSV y un resumen en TXT.

Requisitos:
    pip install -U ultralytics opencv-python pandas pyyaml numpy

Ajusta las rutas de configuración más abajo si hace falta.
"""

import os
import sys
import time
import json
import math
import shutil
import random
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import cv2

# -----------------------
# CONFIGURACIÓN (ajusta)
# -----------------------
BASE_DIR = Path("/home/pablo/Documents/pro_vision/planos")
PLANOS_DIR = BASE_DIR / "all_images"

# INPUT dataset (user: images + labels folders)
DATASET_DIR = PLANOS_DIR# / "dataset"            # <-- Ajusta si tu carpeta se llama de otra forma
DATASET_IMAGES = DATASET_DIR / "images"
DATASET_LABELS = DATASET_DIR / "labels"

# Split output (symlinks)
SPLIT_DIR = PLANOS_DIR / "split"

# Where to store models / training runs (base)
PROJECT_RESULTS_BASE = PLANOS_DIR / "models"    # base dir where entrenamiento_N will be created
PROJECT_RESULTS_BASE.mkdir(parents=True, exist_ok=True)

# YAML and model config
MODEL_NAME = "yolov8s.pt"   # pretrained base
EPOCHS = 50
BATCH = 16
IMGSZ = 512
DEVICE = "cpu"             # "cpu", "0" (GPU 0) or "auto"

# Split ratios (train, val, test)
RATIO_TRAIN = 0.70
RATIO_VAL = 0.20
RATIO_TEST = 0.10

# Reproducibility seed
SPLIT_SEED = 42

# Names list (if you already have a data_trenes.yaml with names, the script will try to reuse it)
NAMES_LIST = [
    "cabina", "salon", "vestibulo", "wc_normal", "bufet",
    "fuelles", "anexo", "bicicletas", "personal", "wc_pmr", "corredor"
]

# IoU evaluation settings
PRED_CONF_THRESHOLD = 0.001   # low so we get all predictions (can be filtered later)
IOU_MATCHING_STRATEGY = "greedy"  # currently greedy matching GT->best pred

# -----------------------
# UTILIDADES
# -----------------------
def list_images(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in Path(img_dir).glob("*") if p.suffix.lower() in exts])

def ensure_dirs():
    if not DATASET_IMAGES.exists():
        raise FileNotFoundError(f"Images folder not found: {DATASET_IMAGES}")
    if not DATASET_LABELS.exists():
        raise FileNotFoundError(f"Labels folder not found: {DATASET_LABELS}")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_RESULTS_BASE.mkdir(parents=True, exist_ok=True)

def normalize_box_coords(x1, y1, x2, y2):
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return int(x1), int(y1), int(x2), int(y2)

def read_yolo_txt_as_boxes(txt_path, img_w, img_h):
    """
    Read YOLO normalized labels and return list of [x1,y1,x2,y2,cls]
    """
    boxes = []
    if not Path(txt_path).exists():
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc = float(parts[1]); yc = float(parts[2]); bw = float(parts[3]); bh = float(parts[4])
            x1 = (xc - bw/2.0) * img_w
            x2 = (xc + bw/2.0) * img_w
            y1 = (yc - bh/2.0) * img_h
            y2 = (yc + bh/2.0) * img_h
            x1, y1, x2, y2 = normalize_box_coords(x1, y1, x2, y2)
            boxes.append([x1, y1, x2, y2, cls])
    return boxes

def compute_iou(boxA, boxB):
    """
    boxA, boxB: [x1,y1,x2,y2]
    returns IoU float
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    boxBArea = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union

# -----------------------
# SPLIT DATASET
# -----------------------
def create_split_symlinks(seed=SPLIT_SEED, ratios=(RATIO_TRAIN, RATIO_VAL, RATIO_TEST)):
    """
    Create reproducible split (symlinks) under SPLIT_DIR:
      split/train/images, split/train/labels, split/val/..., split/test/...
    Returns dict with lists of Paths for train/val/test images.
    """
    imgs = list_images(DATASET_IMAGES)
    N = len(imgs)
    if N == 0:
        raise RuntimeError("No images found in dataset images folder.")

    random.seed(seed)
    indices = list(range(N))
    random.shuffle(indices)

    r_train, r_val, r_test = ratios
    n_train = int(round(N * r_train))
    n_val = int(round(N * r_val))
    # assign remainder to test
    n_test = N - n_train - n_val
    if n_test < 0:
        # adjust
        n_val = max(0, N - n_train)
        n_test = N - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    sets = {
        "train": [imgs[i] for i in train_idx],
        "val": [imgs[i] for i in val_idx],
        "test": [imgs[i] for i in test_idx]
    }

    # create dirs
    for split in ("train", "val", "test"):
        (SPLIT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (SPLIT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # create symlinks (images + corresponding labels if exist)
    for split, imgs_list in sets.items():
        for img_path in imgs_list:
            target_img = img_path.resolve()
            link_img = (SPLIT_DIR / split / "images" / img_path.name)
            if not link_img.exists():
                try:
                    os.symlink(str(target_img), str(link_img))
                except FileExistsError:
                    pass
                except OSError:
                    # fallback to copy if symlink not allowed
                    shutil.copy2(str(target_img), str(link_img))
            # label
            label_src = DATASET_LABELS / (img_path.stem + ".txt")
            link_label = (SPLIT_DIR / split / "labels" / (img_path.stem + ".txt"))
            if label_src.exists() and not link_label.exists():
                try:
                    os.symlink(str(label_src.resolve()), str(link_label))
                except OSError:
                    shutil.copy2(str(label_src), str(link_label))

    return sets

# -----------------------
# DATA YAML writer
# -----------------------
def write_data_yaml_for_split(out_yaml_path, train_dir, val_dir, names):
    data = {
        "train": str(train_dir.resolve()),
        "val": str(val_dir.resolve()),
        "nc": len(names),
        "names": names
    }
    with open(out_yaml_path, "w") as f:
        yaml.safe_dump(data, f)
    return out_yaml_path

# -----------------------
# TRAIN COUNTER
# -----------------------
def read_and_increment_counter(counter_file: Path):
    if not counter_file.exists():
        counter_file.write_text("1")
        return 1
    try:
        v = int(counter_file.read_text().strip())
    except Exception:
        v = 0
    v = v + 1
    counter_file.write_text(str(v))
    return v

# -----------------------
# TRAINING wrapper
# -----------------------
def train_yolo(data_yaml_path: Path, project_dir: Path, run_name: str, epochs=EPOCHS, batch=BATCH, imgsz=IMGSZ, device=DEVICE, base_model=MODEL_NAME):
    """
    Trains YOLOv8 using ultralytics API. Returns path to best.pt (or None).
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics not installed. pip install -U ultralytics") from e

    model = YOLO(base_model)
    project_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TRAIN] Starting training, project dir: {project_dir}, run name: {run_name}")
    t0 = time.time()
    model.train(data=str(data_yaml_path), epochs=epochs, batch=batch, imgsz=imgsz,
                device=device, project=str(project_dir), name=run_name, exist_ok=True)
    t1 = time.time()
    elapsed = t1 - t0
    # find best.pt inside project_dir/run_name/weights/best.pt
    run_dir = project_dir / run_name
    cand = run_dir / "weights" / "best.pt"
    if cand.exists():
        return cand, elapsed
    # fallback: search newest .pt under run_dir
    pts = list(run_dir.rglob("*.pt"))
    if pts:
        pts_sorted = sorted(pts, key=lambda p: p.stat().st_mtime, reverse=True)
        return pts_sorted[0], elapsed
    return None, elapsed

# -----------------------
# EVALUATION on test split using IoU
# -----------------------
def evaluate_model_on_test(model_pt_path: Path, test_images, test_labels_dir: Path, names_list):
    """
    Runs inference on test_images and computes IoU statistics:
      - per-image: mean/min/max IoU over GT boxes
      - per-class: mean IoU over GTs of that class
      - global: mean IoU, std, total GT count, detected preds count
    Returns summary dict and per-image list.
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics not installed. pip install -U ultralytics") from e

    model = YOLO(str(model_pt_path))
    per_image_rows = []
    per_class_ious = {i: [] for i in range(len(names_list))}
    all_ious = []

    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt_boxes = read_yolo_txt_as_boxes(test_labels_dir / (img_path.stem + ".txt"), W, H)  # [x1,y1,x2,y2,cls]
        # Run predictions
        results = model.predict(source=str(img_path), conf=PRED_CONF_THRESHOLD, iou=0.5, show=False, verbose=False)
        preds = []
        for r in results:
            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    x1, y1, x2, y2 = map(float, box.tolist())
                    preds.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), int(cls)])
        # For each GT, compute best IoU among preds
        ious_img = []
        for gt in gt_boxes:
            gt_box = gt[:4]
            best_iou = 0.0
            best_pred = None
            for pr in preds:
                pr_box = pr[:4]
                iou = compute_iou(gt_box, pr_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pr
            ious_img.append(best_iou)
            all_ious.append(best_iou)
            # per-class
            cls_gt = gt[4]
            if 0 <= cls_gt < len(names_list):
                per_class_ious[cls_gt].append(best_iou)
        mean_iou_img = float(np.mean(ious_img)) if len(ious_img) > 0 else 0.0
        min_iou_img = float(np.min(ious_img)) if len(ious_img) > 0 else 0.0
        max_iou_img = float(np.max(ious_img)) if len(ious_img) > 0 else 0.0
        per_image_rows.append({
            "image": img_path.name,
            "n_gt": len(gt_boxes),
            "n_pred": len(preds),
            "mean_iou": mean_iou_img,
            "min_iou": min_iou_img,
            "max_iou": max_iou_img
        })

    # per-class stats
    per_class_summary = {}
    for cls_id, ious in per_class_ious.items():
        if len(ious) > 0:
            per_class_summary[cls_id] = {
                "class_name": names_list[cls_id] if cls_id < len(names_list) else f"class_{cls_id}",
                "mean_iou": float(np.mean(ious)),
                "std_iou": float(np.std(ious)),
                "count": len(ious)
            }
        else:
            per_class_summary[cls_id] = {
                "class_name": names_list[cls_id] if cls_id < len(names_list) else f"class_{cls_id}",
                "mean_iou": None,
                "std_iou": None,
                "count": 0
            }

    global_mean = float(np.mean(all_ious)) if len(all_ious) > 0 else 0.0
    global_std = float(np.std(all_ious)) if len(all_ious) > 0 else 0.0
    total_gt = len(all_ious)

    summary = {
        "global_mean_iou": global_mean,
        "global_std_iou": global_std,
        "total_gt_boxes": total_gt,
        "per_class": per_class_summary
    }

    return summary, per_image_rows

# -----------------------
# MAIN FLOW
# -----------------------
def main(seed=SPLIT_SEED, overwrite_split=False):
    ensure_dirs()

    # 1) Create split (symlinks)
    print("[STEP] Creating splits (train/val/test)...")
    if overwrite_split:
        # remove existing split dir
        if SPLIT_DIR.exists():
            shutil.rmtree(SPLIT_DIR)
    sets = create_split_symlinks(seed=seed)
    train_imgs = sets["train"]
    val_imgs = sets["val"]
    test_imgs = sets["test"]
    print(f" -> train: {len(train_imgs)} images, val: {len(val_imgs)}, test: {len(test_imgs)}")

    # 2) Read/Increment training counter and create output folder for this run
    counter_file = PROJECT_RESULTS_BASE / "train_counter.txt"
    run_number = read_and_increment_counter(counter_file)
    run_name = f"entrenamiento_{run_number}"
    run_project_dir = PROJECT_RESULTS_BASE / run_name
    run_project_dir.mkdir(parents=True, exist_ok=True)
    print(f"[STEP] Training run folder: {run_project_dir}")

    # 3) create data yaml for this split (train uses split/train/images; val uses split/val/images)
    data_yaml_path = run_project_dir / "data_trenes_split.yaml"
    train_images_dir = SPLIT_DIR / "train" / "images"
    val_images_dir = SPLIT_DIR / "val" / "images"
    write_data_yaml_for_split(data_yaml_path, train_images_dir, val_images_dir, NAMES_LIST)
    print("[STEP] Data YAML written to:", data_yaml_path)

    # 4) Train model
    print("[STEP] Training model (this may take several minutes)...")
    try:
        best_pt, elapsed = train_yolo(data_yaml_path, run_project_dir, run_name, epochs=EPOCHS, batch=BATCH, imgsz=IMGSZ, device=DEVICE, base_model=MODEL_NAME)
    except Exception as e:
        print("Training failed:", e)
        return

    print(f"[STEP] Training finished in {elapsed/60.0:.2f} minutes. best_pt: {best_pt}")
    if best_pt is None:
        print("No weights found, aborting evaluation.")
        return

    # copy best.pt into run root for convenience
    try:
        shutil.copy2(str(best_pt), str(run_project_dir / "best.pt"))
    except Exception:
        pass

    # 5) Evaluate on test split using IoU
    print("[STEP] Evaluating best model on test split...")
    test_labels_dir = SPLIT_DIR / "test" / "labels"
    summary, per_image = evaluate_model_on_test(best_pt, test_imgs, test_labels_dir, NAMES_LIST)

    # 6) Save results (CSV + JSON + TXT)
    per_image_df = pd.DataFrame(per_image)
    per_image_csv = run_project_dir / "iou_per_image.csv"
    per_image_df.to_csv(per_image_csv, index=False)

    # per-class summary to CSV
    per_class_rows = []
    for cls_id, info in summary["per_class"].items():
        per_class_rows.append({
            "class_id": cls_id,
            "class_name": info["class_name"],
            "mean_iou": info["mean_iou"],
            "std_iou": info["std_iou"],
            "count": info["count"]
        })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = run_project_dir / "iou_per_class.csv"
    per_class_df.to_csv(per_class_csv, index=False)

    # global summary
    summary_json = run_project_dir / "iou_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    summary_txt = run_project_dir / "iou_summary.txt"
    with open(summary_txt, "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Train images: {len(train_imgs)}, Val images: {len(val_imgs)}, Test images: {len(test_imgs)}\n")
        f.write(f"Training time (s): {elapsed:.2f}\n")
        f.write(f"Best weights: {str(best_pt)}\n\n")
        f.write(f"Global mean IoU: {summary['global_mean_iou']:.6f}\n")
        f.write(f"Global std IoU: {summary['global_std_iou']:.6f}\n")
        f.write(f"Total GT boxes evaluated: {summary['total_gt_boxes']}\n\n")
        f.write("Per-class IoU:\n")
        for cls_id, info in summary["per_class"].items():
            f.write(f" - {info['class_name']} (id {cls_id}): mean={info['mean_iou']} std={info['std_iou']} count={info['count']}\n")

    print("[STEP] Evaluation saved:")
    print(" - per-image CSV:", per_image_csv)
    print(" - per-class CSV:", per_class_csv)
    print(" - summary JSON:", summary_json)
    print(" - summary TXT:", summary_txt)
    print("[DONE] Run completed.")

# -----------------------
# ENTRYPOINT
# -----------------------
if __name__ == "__main__":
    # default behavior: use configured seed, don't overwrite split if exists
    # you can call this script with environment variables to tweak behavior:
    #   SPLIT_SEED, OVERWRITE_SPLIT (1 to force)
    env_seed = os.environ.get("SPLIT_SEED")
    if env_seed is not None:
        try:
            seed_val = int(env_seed)
        except:
            seed_val = SPLIT_SEED
    else:
        seed_val = SPLIT_SEED

    overwrite = os.environ.get("OVERWRITE_SPLIT", "0") in ("1", "true", "True")
    main(seed=seed_val, overwrite_split=overwrite)
