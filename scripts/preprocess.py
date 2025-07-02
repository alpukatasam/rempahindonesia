import os, glob, shutil
from sklearn.model_selection import train_test_split
from PIL import Image

RAW_DIR   = "data/Indonesian Spices Dataset"
SPLIT_DIR = "data_split"
RATIOS    = (0.7, 0.2, 0.1)  # train, val, test
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# 1. Split data
for subset in ["train","val","test"]:
    os.makedirs(os.path.join(SPLIT_DIR, subset), exist_ok=True)

for cls in os.listdir(RAW_DIR):
    src = os.path.join(RAW_DIR, cls)
    if not os.path.isdir(src): continue
    files = glob.glob(os.path.join(src, "*.*"))
    train, temp = train_test_split(files, test_size=1-RATIOS[0], random_state=42)
    val, test  = train_test_split(temp, test_size=RATIOS[2]/(RATIOS[1]+RATIOS[2]), random_state=42)
    for split, flist in zip(["train","val","test"], [train,val,test]):
        dst = os.path.join(SPLIT_DIR, split, cls)
        os.makedirs(dst, exist_ok=True)
        for f in flist:
            shutil.copy(f, dst)

# 2. Hapus file non-gambar atau korup
for subset in ["train","val","test"]:
    base = os.path.join(SPLIT_DIR, subset)
    for cls in os.listdir(base):
        for fname in os.listdir(os.path.join(base, cls)):
            if not fname.lower().endswith(valid_exts):
                os.remove(os.path.join(base, cls, fname))
            else:
                # cek korupsi
                path = os.path.join(base, cls, fname)
                try:
                    Image.open(path).verify()
                except:
                    os.remove(path)

print("✅ Preprocessing selesai")
