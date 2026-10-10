import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. DATASET PATHS
# ============================================================

train_dir = "dataset_split/train"
val_dir = "dataset_split/val"
test_dir = "dataset_split/test"


# ============================================================
# 2. BASIC SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# 4. LOAD VALIDATION DATA
# ============================================================

val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 5. LOAD TEST DATA
# ============================================================

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 6. GET CLASS NAMES
# ============================================================

class_names = train_data.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print("\nNumber of classes:", len(class_names))


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.1),

    tf.keras.layers.RandomZoom(0.1),

    tf.keras.layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    )
])


# ============================================================
# 8. CALCULATE CLASS WEIGHTS
# ============================================================

class_counts = []

for class_name in class_names:

    class_path = os.path.join(
        train_dir,
        class_name
    )

    count = len([
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ])

    class_counts.append(count)


# Create labels for calculating balanced weights
labels = np.concatenate([
    np.full(count, index)
    for index, count in enumerate(class_counts)
])


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=labels
)


class_weights = {
    index: weight
    for index, weight in enumerate(class_weights_array)
}


# ============================================================
# 9. DISPLAY CLASS COUNTS AND WEIGHTS
# ============================================================

print("\nClass counts:")
print("----------------")

for name, count in zip(
    class_names,
    class_counts
):
    print(f"{name}: {count}")


print("\nClass weights:")
print("----------------")

for index, weight in class_weights.items():
    print(
        f"{class_names[index]}: {weight:.3f}"
    )


# ============================================================
# 10. BUILD CUSTOM CNN
# ============================================================

model = tf.keras.Sequential([

    # Input layer
    tf.keras.layers.Input(
        shape=(224, 224, 3)
    ),

    # Data augmentation
    data_augmentation,

    # Normalize pixels
    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    # -------------------------
    # CNN Block 1
    # -------------------------

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),


    # -------------------------
    # CNN Block 2
    # -------------------------

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),


    # -------------------------
    # CNN Block 3
    # -------------------------

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),


    # -------------------------
    # Classification layers
    # -------------------------

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout helps reduce overfitting
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 12. DISPLAY MODEL ARCHITECTURE
# ============================================================

print("\n")
print("=" * 60)
print("CUSTOM CNN MODEL")
print("=" * 60)

model.summary()
