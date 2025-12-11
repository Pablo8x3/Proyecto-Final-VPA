import os
from pathlib import Path
import cv2

# -----------------------------
# CONFIGURE YOUR DIRECTORIES
# -----------------------------
img_dir = Path("planos/all_images/images")
lbl_dir = Path("planos/all_images/labels")

out_img_dir = img_dir#  / "augmented"
out_lbl_dir = lbl_dir#  / "augmented"

out_img_dir.mkdir(exist_ok=True)
out_lbl_dir.mkdir(exist_ok=True)

# ---------------------------------
# FUNCTION TO FLIP YOLO BBOX
# ---------------------------------
def flip_bbox_horizontally(cx, cy, w, h):
    """
    Horizontal flip (mirror left-to-right)

    New center x is: 1 - cx
    y, w, h do NOT change
    """
    new_cx = 1.0 - cx
    return new_cx, cy, w, h

# ---------------------------------
# PROCESS ALL IMAGES
# ---------------------------------
images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

print(f"Found {len(images)} images")

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        print("ERROR loading:", img_path)
        continue

    h, w = img.shape[:2]

    # Flip image horizontally
    flipped = cv2.flip(img, 1)

    # Save flipped image
    out_img_path = out_img_dir / f"{img_path.stem}_flip{img_path.suffix}"
    cv2.imwrite(str(out_img_path), flipped)

    # -----------------------------
    # PROCESS LABELS
    # -----------------------------
    label_file = lbl_dir / f"{img_path.stem}.txt"
    out_label_file = out_lbl_dir / f"{img_path.stem}_flip.txt"

    if not label_file.exists():
        # Some images may not have labels
        open(out_label_file, "w").close()
        continue

    new_lines = []

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            # Flip horizontally
            cx_new, cy_new, bw_new, bh_new = flip_bbox_horizontally(cx, cy, bw, bh)

            new_lines.append(
                f"{cls_id} {cx_new:.6f} {cy_new:.6f} {bw_new:.6f} {bh_new:.6f}"
            )

    # Save flipped labels
    with open(out_label_file, "w") as f:
        f.write("\n".join(new_lines))

    print("Flipped:", img_path.name)

print("\n✅ Done! Augmented images saved in:")
print(out_img_dir)
print("Labels saved in:")
print(out_lbl_dir)
