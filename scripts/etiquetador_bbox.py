import os
import cv2
import numpy as np

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = os.path.join(BASE_DIR, "..", "planos", "train")
LABELS_DIR = os.path.join(IMAGES_DIR, "labels")

os.makedirs(LABELS_DIR, exist_ok=True)

# --- CLASES ---
CLASSES = [
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

# --- COLORES (borde + relleno RGBA) ---
COLOR_MAP = {
    "cabina":        {"edge": (0, 100, 0),      "fill": (0, 255, 0, 80)},
    "salon":         {"edge": (100, 0, 0),      "fill": (255, 0, 0, 80)},
    "vestibulo":     {"edge": (0, 120, 255),    "fill": (0, 180, 255, 80)},
    "wc_normal":     {"edge": (0, 0, 150),      "fill": (0, 0, 255, 80)},
    "bufet":         {"edge": (0, 180, 180),    "fill": (0, 255, 255, 80)},
    "fuelles":       {"edge": (150, 0, 150),    "fill": (200, 0, 200, 80)},
    "anexo":         {"edge": (80, 80, 80),     "fill": (160, 160, 160, 80)},
    "bicicletas":    {"edge": (200, 80, 200),   "fill": (230, 130, 230, 80)},
    "personal":      {"edge": (100, 50, 0),     "fill": (150, 80, 0, 80)},
    "wc_pmr":        {"edge": (0, 150, 80),     "fill": (0, 255, 150, 80)},
    "corredor":      {"edge": (255, 150, 0),    "fill": (255, 200, 50, 80)}
}

drawing = False
ix, iy = -1, -1
boxes = []
img = None
img_draw = None


def seleccionar_clase():
    print("\nSelecciona una clase:")
    for i, c in enumerate(CLASSES):
        print(f"{i}: {c}")

    while True:
        k = input("Clase: ")
        if k.isdigit() and int(k) in range(len(CLASSES)):
            return int(k)
        print("Clase inválida.")


def normalizar_bbox(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h
    return cx, cy, bw, bh


def dibujar_caja(img_base, x1, y1, x2, y2, clase):
    color_edge = COLOR_MAP[CLASSES[clase]]["edge"]
    color_fill = COLOR_MAP[CLASSES[clase]]["fill"]

    overlay = img_base.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_fill[:3], -1)

    alpha = color_fill[3] / 255.0
    img_out = cv2.addWeighted(overlay, alpha, img_base, 1 - alpha, 0)

    cv2.rectangle(img_out, (x1, y1), (x2, y2), color_edge, 2)
    return img_out


def redibujar_todas():
    global img_draw
    img_draw = img.copy()
    for (x1, y1, x2, y2, clase) in boxes:
        img_draw = dibujar_caja(img_draw, x1, y1, x2, y2, clase)


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, img_draw, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_draw = img.copy()
            redibujar_todas()
            cv2.rectangle(img_draw, (ix, iy), (x, y), (255, 255, 255), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        clase = seleccionar_clase()
        boxes.append((ix, iy, x, y, clase))
        redibujar_todas()


def guardar_etiquetas(img_name, w, h):
    name_txt = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(LABELS_DIR, name_txt)

    with open(label_path, "w") as f:
        for (x1, y1, x2, y2, clase) in boxes:
            cx, cy, bw, bh = normalizar_bbox(x1, y1, x2, y2, w, h)
            f.write(f"{clase} {cx} {cy} {bw} {bh}\n")

    print(f"Guardado: {label_path}")


# -------------------------
#         MAIN
# -------------------------

imagenes = [f for f in os.listdir(IMAGES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for img_name in imagenes:
    boxes = []

    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"No se pudo cargar {img_name}")
        continue

    img_draw = img.copy()
    cv2.namedWindow("ETIQUETADOR")
    cv2.setMouseCallback("ETIQUETADOR", mouse_callback)

    print(f"\nImagen: {img_name}")
    print("Instrucciones:")
    print(" - Click y arrastrar para dibujar")
    print(" - Suelta → se pide clase")
    print(" - 'u' → deshacer última caja")
    print(" - 'n' → guardar y siguiente imagen")
    print(" - 'q' → salir sin guardar")

    while True:
        cv2.imshow("ETIQUETADOR", img_draw)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('u'):
            if boxes:
                boxes.pop()
                redibujar_todas()
            print("Última caja eliminada.")

        if key == ord('n'):
            h, w = img.shape[:2]
            guardar_etiquetas(img_name, w, h)
            break

        if key == ord('q'):
            exit()

    cv2.destroyAllWindows()

print("\nEtiquetado completado.")
