import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# ==========================================
# SETTINGS
# ==========================================

test_dir = "dataset_split/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ==========================================
# LOAD TEST DATA
# ==========================================

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_data.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# ==========================================
# LOAD SAVED MOBILENETV2
# ==========================================

print("\nLoading saved MobileNetV2 model...")

model = tf.keras.models.load_model(
    "models/mobilenetv2_best.keras"
)

print("Model loaded successfully!")

# ==========================================
# TEST EVALUATION
# ==========================================

print("\n" + "=" * 60)
print("MOBILENETV2 TEST RESULTS")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_data)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# ==========================================
# GET TRUE LABELS
# ==========================================

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_data
])

# ==========================================
# GET PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

y_prob = model.predict(test_data)

y_pred = np.argmax(y_prob, axis=1)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# ==========================================
# PLOT CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(12, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Disease")
plt.ylabel("Actual Disease")
plt.title("MobileNetV2 Confusion Matrix")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "models/mobilenetv2_confusion_matrix.png",
    dpi=300
)

plt.show()