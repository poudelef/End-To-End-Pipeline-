# src/data_preprocessing.py
import tensorflow as tf
import numpy as np
from pathlib import Path
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def preprocess_data(dataset_root: Path):
    """
    dataset_root: path to folder that contains 'train' and 'test'
    """
    train_dir = dataset_root / "train"
    test_dir  = dataset_root / "test"

    print("Train dir:", train_dir)
    print("Test dir:", test_dir)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    num_classes = train_ds.element_spec[1].shape[-1]

    return train_ds, test_ds, num_classes

def class_names(train_dir: Path):
    data_dir = Path(train_dir)
    return np.array(sorted([item.name for item in data_dir.glob('*') if item.is_dir()]))

def count_images_in_subfolders(root_directory: Path):
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_counts = {}
    for dirpath, _, filenames in os.walk(root_directory):
        count = sum(1 for f in filenames if f.lower().endswith(image_extensions))
        image_counts[dirpath] = count
    return image_counts
