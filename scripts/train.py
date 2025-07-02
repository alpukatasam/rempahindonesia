import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Konfigurasi
DATA_DIR   = "data_split"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCH_HEAD = 10
EPOCH_FINE = 5

# Fungsi buat dataset
def make_ds(subset, augment=False):
    path = os.path.join(DATA_DIR, subset)
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, label_mode="categorical"
    )
    ds = ds.map(lambda x,y: (preprocess_input(x), y), tf.data.AUTOTUNE)
    if augment:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        ds = ds.map(lambda x,y: (aug(x), y), tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

# Load dataset
train_ds = make_ds("train", augment=True)
val_ds   = make_ds("val")

# Print class order
print("Class names:", train_ds.class_names)

# Bangun model
base = EfficientNetV2S(include_top=False, weights="imagenet",
                       input_shape=(*IMG_SIZE, 3))
base.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.2)(x)
out = layers.Dense(train_ds.element_spec[1].shape[-1],
                   activation="softmax")(x)
model = Model(base.input, out)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
ckpt = ModelCheckpoint("models/best_model.h5", save_best_only=True,
                       monitor="val_accuracy")
es   = EarlyStopping(patience=3, restore_best_weights=True)

# Latih head
history_head = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCH_HEAD, callbacks=[ckpt, es]
)

# Fine-tuning
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCH_FINE, callbacks=[ckpt, es]
)

# Plot akurasi
plt.plot(history_head.history["val_accuracy"], label="Head")
plt.plot(history_fine.history["val_accuracy"], label="Fine-tune")
plt.legend(); plt.show()

valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
for split in ["train", "val", "test"]:
    split_dir = os.path.join(SPLIT_DIR, split)
    for cls in os.listdir(split_dir):
        cls_dir = os.path.join(split_dir, cls)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(valid_exts):
                path = os.path.join(cls_dir, fname)
                os.remove(path)
                print("Removed:", path)
print("✅ Pembersihan file non-gambar selesai")