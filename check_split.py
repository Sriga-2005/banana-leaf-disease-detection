import os

dataset_dir = "dataset_split"

for split in ["train", "val", "test"]:

    print(f"\n{split.upper()}")
    print("-" * 40)

    total = 0

    split_path = os.path.join(dataset_dir, split)

    for class_name in sorted(os.listdir(split_path)):

        class_path = os.path.join(
            split_path,
            class_name
        )

        if not os.path.isdir(class_path):
            continue

        images = [
            file
            for file in os.listdir(class_path)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ]

        print(f"{class_name}: {len(images)}")

        total += len(images)

    print(f"Total: {total}")