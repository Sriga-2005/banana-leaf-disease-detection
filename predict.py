import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/mobilenetv2_best.keras"

TEST_DIR = "dataset_split/test"

IMG_SIZE = (224, 224)

# ==========================================
# CLASS NAMES
# ==========================================

class_names = [
    "Banana Black Sigatoka Disease",
    "Banana Bract Mosaic Virus Disease",
    "Banana Healthy Leaf",
    "Banana Insect Pest Disease",
    "Banana Moko Disease",
    "Banana Panama Disease",
    "Banana Yellow Sigatoka Disease"
]

# ==========================================
# FIND AN IMAGE AUTOMATICALLY
# ==========================================

image_path = None

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)

for class_name in class_names:

    class_folder = os.path.join(
        TEST_DIR,
        class_name
    )

    if os.path.exists(class_folder):

        for filename in os.listdir(class_folder):

            if filename.lower().endswith(valid_extensions):

                image_path = os.path.join(
                    class_folder,
                    filename
                )

                break

    if image_path is not None:
        break

# ==========================================
# CHECK IMAGE
# ==========================================

if image_path is None:

    print("ERROR: No image found in the test dataset.")

    print("\nPlease check:")
    print("dataset_split/test/")

    exit()

print("\nImage selected:")
print(image_path)

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading MobileNetV2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")

# ==========================================
# LOAD IMAGE
# ==========================================

image = load_img(
    image_path,
    target_size=IMG_SIZE
)

image_array = img_to_array(image)

# Add batch dimension
image_array = np.expand_dims(
    image_array,
    axis=0
)

# ==========================================
# MOBILENETV2 PREPROCESSING
# ==========================================

image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    image_array
)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(
    prediction[0]
)

predicted_class = class_names[
    predicted_index
]

confidence = (
    prediction[0][predicted_index] * 100
)

# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n")
print("=" * 60)
print("BANANA LEAF DISEASE PREDICTION")
print("=" * 60)

print("\nInput Image:")
print(image_path)

print("\nPredicted Disease:")
print(predicted_class)

print(f"\nConfidence: {confidence:.2f}%")

# ==========================================
# ALL CLASS PROBABILITIES
# ==========================================

print("\nClass Probabilities:")
print("-" * 60)

for name, probability in zip(
    class_names,
    prediction[0]
):

    print(
        f"{name:<40} : {probability * 100:.2f}%"
    )

print("=" * 60)