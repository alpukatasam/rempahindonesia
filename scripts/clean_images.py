import os
from PIL import Image

# Path ke hasil split
SPLIT_DIR = "data_split"
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

bad_files = []

for subset in ["train", "val", "test"]:
    subset_dir = os.path.join(SPLIT_DIR, subset)
    for cls in os.listdir(subset_dir):
        cls_dir = os.path.join(subset_dir, cls)
        for fname in os.listdir(cls_dir):
            path = os.path.join(cls_dir, fname)
            # 1) ekstensi tidak valid
            if not fname.lower().endswith(valid_exts):
                bad_files.append(path)
                os.remove(path)
                continue
            # 2) cek korupsi
            try:
                Image.open(path).verify()
            except Exception:
                bad_files.append(path)
                os.remove(path)

print(f"Removed {len(bad_files)} bad files:")
for p in bad_files:
    print(" -", p)
