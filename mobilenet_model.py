import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# SETTINGS
# ==========================================

train_dir = "dataset_split/train"
val_dir = "dataset_split/val"
test_dir = "dataset_split/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 20

os.makedirs("models", exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_data.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print("\nNumber of classes:", len(class_names))

# ==========================================
# CLASS WEIGHTS
# ==========================================

class_counts = []

for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)

    count = len([
        file for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ])

    class_counts.append(count)

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

print("\nClass weights:")
print("----------------")

for index, weight in class_weights.items():
    print(f"{class_names[index]}: {weight:.3f}")

# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    )
])

# ==========================================
# MOBILENETV2 BASE MODEL
# ==========================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# ==========================================
# BUILD MODEL
# ==========================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n" + "=" * 60)
print("MOBILENETV2 MODEL")
print("=" * 60)

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/mobilenetv2_best.keras",
    monitor="val_loss",
    save_best_only=True
)

# ==========================================
# TRAINING
# ==========================================

print("\n" + "=" * 60)
print("STARTING MOBILENETV2 TRAINING")
print("=" * 60)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)

# ==========================================
# TRAINING GRAPHS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Accuracy")

plt.legend()
plt.grid(True)

plt.savefig(
    "models/mobilenetv2_accuracy.png",
    dpi=300
)

plt.show()

# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    "models/mobilenetv2_loss.png",
    dpi=300
)

plt.show()

# ==========================================
# TEST EVALUATION
# ==========================================

print("\n" + "=" * 60)
print("MOBILENETV2 TEST RESULTS")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_data)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")