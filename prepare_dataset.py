import os
import shutil
from sklearn.model_selection import train_test_split

# -----------------------------------
# 1. Paths
# -----------------------------------

source_dir = "dataset/Banana_leaf_dieases_dataset/Original Images"
output_dir = "dataset_split"


# -----------------------------------
# 2. Create output folders
# -----------------------------------

classes = [
    folder
    for folder in os.listdir(source_dir)
    if os.path.isdir(os.path.join(source_dir, folder))
]

for split in ["train", "val", "test"]:

    for class_name in classes:

        folder_path = os.path.join(
            output_dir,
            split,
            class_name
        )

        os.makedirs(
            folder_path,
            exist_ok=True
        )


# -----------------------------------
# 3. Split each class
# -----------------------------------

for class_name in classes:

    class_path = os.path.join(
        source_dir,
        class_name
    )

    images = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    # First: 70% train, 30% temporary
    train_images, temp_images = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    # Then: split the 30% into 15% validation and 15% test
    val_images, test_images = train_test_split(
        temp_images,
        test_size=0.50,
        random_state=42
    )

    # Copy training images
    for image in train_images:

        shutil.copy2(
            os.path.join(class_path, image),
            os.path.join(
                output_dir,
                "train",
                class_name,
                image
            )
        )

    # Copy validation images
    for image in val_images:

        shutil.copy2(
            os.path.join(class_path, image),
            os.path.join(
                output_dir,
                "val",
                class_name,
                image
            )
        )

    # Copy testing images
    for image in test_images:

        shutil.copy2(
            os.path.join(class_path, image),
            os.path.join(
                output_dir,
                "test",
                class_name,
                image
            )
        )

    print(
        f"{class_name}: "
        f"{len(train_images)} train, "
        f"{len(val_images)} validation, "
        f"{len(test_images)} test"
    )


print("\nDataset split completed!")