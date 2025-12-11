import cv2
import os
from pathlib import Path

# -----------------------------
# CONFIGURACIÓN
# -----------------------------

IMAGE_DIR = "planos/all_images/images"
LABEL_DIR = "planos/all_images/labels"

CLASSES = [
    "cabina","salon","vestibulo","wc_normal","bufet","fuelles",
    "anexo","bicicletas","personal","wc_pmr","corredor"
]

COLORS = [
    (0,255,0),(255,0,0),(0,140,255),(0,0,255),(0,255,255),
    (255,0,255),(128,128,128),(255,180,140),(139,0,0),
    (50,205,50),(235,206,135)
]

# -----------------------------
# VARIABLES GLOBALES
# -----------------------------
current_box = -1
dragging_corner = None
creating = False
create_start = None
boxes = []  # (cls, x1, y1, x2, y2)
current_class = 0

def normalize_box(x1, y1, x2, y2):
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

# -----------------------------
# Cargar y guardar YOLO
# -----------------------------
def load_yolo_labels(txt_file, w, h):
    data = []
    if not os.path.exists(txt_file):
        return data

    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, xc, yc, bw, bh = map(float, line.split())
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        data.append([int(cls), x1, y1, x2, y2])

    return data


def save_yolo_labels(txt_file, boxes, w, h):
    with open(txt_file, "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


# -----------------------------
# Dibujo
# -----------------------------
def draw_all(img):
    out = img.copy()
    for i, (cls, x1, y1, x2, y2) in enumerate(boxes):
        color = COLORS[cls]
        thickness = 2 if i == current_box else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{CLASSES[cls]}"
        cv2.putText(out, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


# -----------------------------
# Ratón
# -----------------------------
def mouse_callback(event, x, y, flags, param):
    global current_box, dragging_corner, creating, create_start, boxes

    # ----------------- CREACIÓN DE CAJAS -----------------
    if creating:
        if event == cv2.EVENT_LBUTTONDOWN:
            if create_start is None:
                create_start = (x, y)
            else:
                x1, y1 = create_start
                x2, y2 = x, y
                x1, y1, x2, y2 = normalize_box(x1, y1, x2, y2)
                boxes.append([current_class, x1, y1, x2, y2])
                print(f"[OK] Creada bbox clase {CLASSES[current_class]}")
                creating = False
                create_start = None
        return

    # ----------------- SELECCIÓN DE CAJAS -----------------
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_box = -1
        smallest_area = 1e18

        for i, (cls, x1, y1, x2, y2) in enumerate(boxes):
            x1n, y1n, x2n, y2n = normalize_box(x1, y1, x2, y2)

            if x1n <= x <= x2n and y1n <= y <= y2n:
                area = abs((x2n - x1n) * (y2n - y1n))
                if area < smallest_area:
                    smallest_area = area
                    clicked_box = i

        if clicked_box != -1:
            current_box = clicked_box
            cls, x1, y1, x2, y2 = boxes[current_box]
            x1, y1, x2, y2 = normalize_box(x1, y1, x2, y2)
            boxes[current_box] = [cls, x1, y1, x2, y2]

            dragging_corner = None
            corner_tolerance = 15

            if abs(x - x1) < corner_tolerance and abs(y - y1) < corner_tolerance: dragging_corner = "tl"
            elif abs(x - x2) < corner_tolerance and abs(y - y1) < corner_tolerance: dragging_corner = "tr"
            elif abs(x - x1) < corner_tolerance and abs(y - y2) < corner_tolerance: dragging_corner = "bl"
            elif abs(x - x2) < corner_tolerance and abs(y - y2) < corner_tolerance: dragging_corner = "br"

            print(f"[OK] Seleccionada bbox {current_box} ({CLASSES[cls]})")
            return

        current_box = -1

    # ----------------- ARRASTRAR ESQUINAS -----------------
    if event == cv2.EVENT_MOUSEMOVE and dragging_corner is not None and current_box != -1:
        cls, x1, y1, x2, y2 = boxes[current_box]

        if dragging_corner == "tl": x1, y1 = x, y
        if dragging_corner == "tr": x2, y1 = x, y
        if dragging_corner == "bl": x1, y2 = x, y
        if dragging_corner == "br": x2, y2 = x, y

        x1, y1, x2, y2 = normalize_box(x1, y1, x2, y2)
        boxes[current_box] = [cls, x1, y1, x2, y2]

    if event == cv2.EVENT_LBUTTONUP:
        dragging_corner = None


# -----------------------------
# Main
# -----------------------------
def main():
    global boxes, current_box, creating, current_class

    images = sorted(Path(IMAGE_DIR).glob("*.jpg"))
    idx = 0

    while True:
        img_path = str(images[idx])
        label_path = str(Path(LABEL_DIR) / (Path(img_path).stem + ".txt"))

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        boxes = load_yolo_labels(label_path, w, h)
        current_box = -1
        creating = False

        cv2.namedWindow("BBox Editor")
        cv2.setMouseCallback("BBox Editor", mouse_callback)

        while True:
            display = draw_all(img)
            cv2.imshow("BBox Editor", display)
            key = cv2.waitKey(30)

            if key == ord('q'):
                cv2.destroyAllWindows()
                return

            # -------- CAMBIAR CLASE --------
            if current_box != -1:
                # --- Teclas 0-9 ---
                if key in range(ord('0'), ord('9') + 1):
                    cls_num = key - ord('0')
                    if cls_num < len(CLASSES):
                        boxes[current_box][0] = cls_num
                        current_class = cls_num
                        print(f"Clase cambiada a {CLASSES[cls_num]}")

                # --- Teclas para clases especiales ---
                elif key == ord('p'):  # asignar 'p' a "corredor"
                    cls_num = CLASSES.index("corredor")
                    boxes[current_box][0] = cls_num
                    current_class = cls_num
                    print(f"Clase cambiada a {CLASSES[cls_num]}")

            # -------- ELIMINAR CAJA --------
            if key in [8, 127]:  # Backspace / Supr
                if current_box != -1:
                    print("Caja eliminada")
                    boxes.pop(current_box)
                    current_box = -1

            # -------- CREAR NUEVA CAJA --------
            if key == ord('c'):
                print("Creando nueva bbox: clic inicial y clic final")
                creating = True
                current_box = -1
                create_start = None

            # -------- SIGUIENTE IMAGEN --------
            if key == ord('n'):
                save_yolo_labels(label_path, boxes, w, h)
                idx = (idx + 1) % len(images)
                break

            # -------- ANTERIOR IMAGEN --------
            if key == ord('b'):
                save_yolo_labels(label_path, boxes, w, h)
                idx = (idx - 1) % len(images)
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
