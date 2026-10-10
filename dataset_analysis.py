import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------------
# 1. Dataset path
# -----------------------------------

dataset_path = "dataset/Banana_leaf_dieases_dataset/Original Images"


# -----------------------------------
# 2. Find all class folders
# -----------------------------------

classes = [
    folder
    for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
]


# -----------------------------------
# 3. Count images
# -----------------------------------

print("Classes found:")
print("----------------")

total_images = 0

class_counts = {}

for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    images = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    count = len(images)

    class_counts[class_name] = count
    total_images += count

    print(f"{class_name}: {count} images")


print("----------------")
print(f"Total images: {total_images}")


# -----------------------------------
# 4. Class distribution graph
# -----------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    class_counts.keys(),
    class_counts.values()
)

plt.title("Banana Leaf Disease - Class Distribution")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()
plt.show()


# -----------------------------------
# 5. Display sample images
# -----------------------------------

samples_per_class = 5

fig, axes = plt.subplots(
    len(classes),
    samples_per_class,
    figsize=(15, 18)
)

for row, class_name in enumerate(classes):

    class_path = os.path.join(
        dataset_path,
        class_name
    )

    images = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    selected_images = random.sample(
        images,
        min(samples_per_class, len(images))
    )

    for col in range(samples_per_class):

        ax = axes[row, col]

        if col < len(selected_images):

            image_path = os.path.join(
                class_path,
                selected_images[col]
            )

            image = Image.open(image_path)

            ax.imshow(image)
            ax.axis("off")

        else:
            ax.axis("off")

        if col == 0:
            ax.set_title(
                class_name,
                fontsize=9,
                loc="left"
            )

plt.tight_layout()
plt.show()
# -----------------------------------
# 6. Check image dimensions
# -----------------------------------

from PIL import Image

print("\nIMAGE QUALITY CHECK")
print("-------------------")

dimension_counts = {}
corrupted_images = []

for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    images = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    for file in images:

        image_path = os.path.join(class_path, file)

        try:
            with Image.open(image_path) as image:

                # Check whether image is valid
                image.verify()

            # Open again after verify()
            with Image.open(image_path) as image:

                width, height = image.size

                dimensions = (width, height)

                dimension_counts[dimensions] = (
                    dimension_counts.get(dimensions, 0) + 1
                )

        except Exception as e:

            corrupted_images.append(
                (class_name, file)
            )


# -----------------------------------
# 7. Print dimension results
# -----------------------------------

print("\nImage dimensions found:")
print("-----------------------")

for dimensions, count in sorted(
    dimension_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{dimensions[0]} x {dimensions[1]} : {count} images"
    )


# -----------------------------------
# 8. Print corrupted images
# -----------------------------------

print("\nCorrupted images:")
print("-----------------")

if len(corrupted_images) == 0:

    print("No corrupted images found.")

else:

    for class_name, file in corrupted_images:

        print(
            f"{class_name} -> {file}"
        )

print(
    f"\nTotal corrupted images: "
    f"{len(corrupted_images)}"
)