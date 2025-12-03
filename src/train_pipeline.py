# src/train_pipeline.py
from pathlib import Path
from data_ingestion import load_data
from data_preprocessing import preprocess_data, class_names, count_images_in_subfolders
from create_model import build_and_train_resnet50
import os
import tensorflow as tf

def main():
    data_url = (
        "https://github.com/poudelef/Datasets/raw/refs/heads/main/"
        "DeepLearning_data_2type_auto-20251122T031133Z-1-001.zip?download="
    )

    # 1. Ingestion
    dataset_root = load_data(data_url)

    # 2. Preprocessing
    train_ds, test_ds, num_classes = preprocess_data(dataset_root)

    print("Detected classes:", class_names(dataset_root / "train"))
    print("Image counts train:", count_images_in_subfolders(dataset_root / "train"))
    print("Image counts test:", count_images_in_subfolders(dataset_root / "test"))

    # 3. Modeling
    model, histories = build_and_train_resnet50(
        train_ds=train_ds,
        val_ds=test_ds,
        num_classes=num_classes
    )

    # 4. Save model
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "resnet50.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # 4. Evaluate model
    Model = tf.keras.models.load_model(model_path, compile=False)
    print("Evaluating loaded model:")
    print(model.evaluate(test_ds))

if __name__ == "__main__":
    main()
