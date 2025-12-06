"""
scripts/train_yolov8.py

Requisitos:
    pip install -U ultralytics

Asume la siguiente estructura (a partir de la carpeta del proyecto):
proyecto/
├─ scripts/
│   └─ train_yolov8.py  <- este archivo
└─ planos/
   ├─ train/            <- imágenes a etiquetar y etiquetadas (labels en train/labels/)
   │   ├─ img1.jpg
   │   └─ labels/
   │       └─ img1.txt
   └─ val/
       ├─ img_val1.jpg
       └─ labels/
           └─ img_val1.txt

El script crea planos/data_trenes.yaml y guarda pesos en planos/models/.
"""

import os
import yaml
from pathlib import Path

# --- Parámetros del entrenamiento (ajusta si quieres) ---
MODEL_NAME = "yolov8s.pt"       # modelo preentrenado base (n, s, m, l, x según necesidad)
EPOCHS = 50
BATCH = 16
IMGSZ = 640
DEVICE = "cpu"                # "cpu", "0" (GPU 0) o "auto"
PROJECT_RESULTS_DIR = "/home/pablo/Documents/pro_vision/planos/models/yolo_trenes"  # desde scripts/ -> ../planos/models

# --- Rutas relativas (no cambies salvo que acomodes tu estructura) ---
BASE_DIR = Path(__file__).resolve().parent
PLANOS_DIR = (BASE_DIR / ".." / "planos").resolve()
TRAIN_DIR = PLANOS_DIR / "train"
VAL_DIR = PLANOS_DIR / "val"
DATA_YAML_PATH = PLANOS_DIR / "data_trenes.yaml"
NAMES_LIST = [
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

def check_dataset_structure():
    # Comprobaciones básicas
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"No existe {TRAIN_DIR}. Coloca las imágenes de entrenamiento ahí.")
    if not VAL_DIR.exists():
        print(f"Advertencia: no existe {VAL_DIR}. Se puede entrenar sin validación, pero es recomendable tener val/ .")
    # Labels
    train_labels = TRAIN_DIR / "labels"
    val_labels = VAL_DIR / "labels"
    if not train_labels.exists():
        print(f"Advertencia: no existe {train_labels}. Asegúrate de que tus .txt de etiquetas estén en train/labels/")
    if not val_labels.exists():
        print(f"Advertencia: no existe {val_labels} (val/labels).")

def write_data_yaml(yaml_path: Path):
    """
    Crea un data YAML para Ultralytics YOLOv8.
    Observa que 'train' y 'val' apuntan a las carpetas que contienen las IMÁGENES.
    La librería buscará los .txt correspondientes en paths paralelos o en labels/.
    """
    data = {
        "train": str(TRAIN_DIR),   # apuntamos a planos/train (contiene images y labels/)
        "val": str(VAL_DIR),
        "nc": len(NAMES_LIST),
        "names": NAMES_LIST
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    print(f"Data YAML escrito en: {yaml_path}")

def train():
    # Import local para que falle aquí si no está ultralytics
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "No se pudo importar 'ultralytics'. Instálalo con: pip install -U ultralytics"
        ) from e

    check_dataset_structure()
    write_data_yaml(DATA_YAML_PATH)

    os.makedirs((PLANOS_DIR / "models"), exist_ok=True)

    # Cargar modelo base (pretrained)
    print(f"Cargando modelo base: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # Ejecutar entrenamiento
    print("Iniciando entrenamiento. Esto puede tardar.")
    results = model.train(
        data=str(DATA_YAML_PATH),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        project=str(PROJECT_RESULTS_DIR),  # donde se guarda run*/
        name="yolo_trenes",
        exist_ok=True  # sobrescribe si hay runs con el mismo nombre
    )

    print("Entrenamiento finalizado. Resultados en:", PROJECT_RESULTS_DIR)
    return results

if __name__ == "__main__":
    train()
