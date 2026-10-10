import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -------------------------------
# SETTINGS
# -------------------------------
test_dir = "dataset_split/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# -------------------------------
# LOAD TEST DATA
# -------------------------------
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_data.class_names

# -------------------------------
# LOAD SAVED MODEL
# -------------------------------
print("\nLoading saved Custom CNN...")

model = tf.keras.models.load_model(
    "models/custom_cnn_best.keras"
)

print("Model loaded successfully!")

# -------------------------------
# TEST ACCURACY
# -------------------------------
print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(test_data)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# -------------------------------
# PREDICTIONS
# -------------------------------
y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_data
])

predictions = model.predict(test_data)

y_pred = np.argmax(predictions, axis=1)

# -------------------------------
# CLASSIFICATION REPORT
# -------------------------------
print("\nCLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_true, y_pred)

print("\nCONFUSION MATRIX")
print("=" * 60)
print(cm)

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
plt.title("Custom CNN - Confusion Matrix")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "models/custom_cnn_confusion_matrix.png",
    dpi=300
)

plt.show()