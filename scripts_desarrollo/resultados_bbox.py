# scripts/test_yolov8_colored.py
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------
# Configuración
# ----------------------
model_path = "planos/models/yolo_trenes/weights/best.pt"  # ruta al modelo entrenado
img_folder = "planos/comprobar_manual/images"                           # carpeta de imágenes a probar
output_folder = "planos/comprobar_manual/results"                           # carpeta donde se guardan resultados
os.makedirs(output_folder, exist_ok=True)

# Clases y colores (RGBA)
class_colors = {
    "cabina": (0, 255, 0, 0.3),         # verde
    "salon": (0, 0, 255, 0.3),          # azul
    "vestibulo": (0, 165, 255, 0.3),    # naranja
    "wc_normal": (0, 0, 255, 0.3),      # rojo
    "fuelles": (128, 0, 128, 0.3),      # morado
    "bufet": (0, 255, 255, 0.3),        # amarillo
    "anexo": (128, 128, 128, 0.3),      # gris
    "bicicletas": (180, 128, 255, 0.3), # morado claro
    "personal": (0, 0, 139, 0.3),       # azul oscuro
    "wc_pmr": (50, 205, 50, 0.3),       # verde lima
    "corredor": (135, 206, 235, 0.3)    # azul celeste
}

# ----------------------
# Función para dibujar bbox
# ----------------------
def draw_bbox(img, box, class_name, conf):
    x1, y1, x2, y2 = map(int, box)
    color_bgr = class_colors[class_name][:3]
    alpha = class_colors[class_name][3]

    # Capa para transparencia
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Borde más oscuro
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

    # Texto
    cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

# ----------------------
# Cargar modelo
# ----------------------
model = YOLO(model_path)

# ----------------------
# Procesar imágenes
# ----------------------
for img_file in Path(img_folder).glob("*.jpg"):
    img_path = str(img_file)
    results = model.predict(source=img_path, conf=0.25, show=False)
    img = cv2.imread(img_path)

    bbox_txt_lines = []

    for r in results:
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            cls_id = int(cls_id)
            conf = float(conf)
            class_name = model.names[cls_id]

            # Guardar bbox en txt (YOLOv8 style: class x_center y_center w h, normalized)
            x1, y1, x2, y2 = box.tolist()
            h, w = img.shape[:2]
            xc = ((x1 + x2)/2) / w
            yc = ((y1 + y2)/2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            bbox_txt_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            # Dibujar bbox en imagen
            draw_bbox(img, (x1, y1, x2, y2), class_name, conf)

    # Guardar txt
    txt_file = Path(output_folder) / f"{img_file.stem}.txt"
    with open(txt_file, "w") as f:
        f.write("\n".join(bbox_txt_lines))

    # Guardar imagen con bbox
    out_img_file = Path(output_folder) / img_file.name
    cv2.imwrite(str(out_img_file), img)

    print(f"Procesada: {img_file.name}, {len(bbox_txt_lines)} detecciones")

print("✅ Procesamiento completo. Resultados en:", output_folder)
