import os
import random


def split_data(data_dir, train_ratio=0.8, test_ratio=0.2, val_ratio=0.0, seed=None):
    if seed is not None:
        random.seed(seed)

    # Get all categories
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # Initialize train, test, and val directories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if val_ratio > 0.0 and not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Process each category
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        image_files = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]

        # Shuffle images
        random.shuffle(image_files)

        # Calculate split counts
        train_count = int(len(image_files) * train_ratio)
        test_count = int(len(image_files) * test_ratio)
        val_count = len(image_files) - train_count - test_count

        # Split into train, test, and val lists
        train_files = image_files[:train_count]
        test_files = image_files[train_count:train_count + test_count]
        val_files = image_files[train_count + test_count:]

        # Ensure each category directory exists in train, test, and val directories
        train_category_dir = os.path.join(train_dir, category)
        test_category_dir = os.path.join(test_dir, category)
        val_category_dir = os.path.join(val_dir, category)

        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(test_category_dir, exist_ok=True)
        if val_ratio > 0.0:
            os.makedirs(val_category_dir, exist_ok=True)

        # Move files to the appropriate directories
        for file in train_files:
            os.rename(os.path.join(category_dir, file), os.path.join(train_category_dir, file))
        for file in test_files:
            os.rename(os.path.join(category_dir, file), os.path.join(test_category_dir, file))
        if val_ratio > 0.0:
            for file in val_files:
                os.rename(os.path.join(category_dir, file), os.path.join(val_category_dir, file))


# Use the function
split_data("..\\datasets\\caltech101", train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, seed=42)
