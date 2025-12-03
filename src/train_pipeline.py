from pathlib import Path
from data_ingestion import load_data
from data_preprocessing import preprocess_data, class_names, count_images_in_subfolders
from create_model import build_and_train_resnet50
import tensorflow as tf

def main():

    # 1. Download or load dataset
    data_url = (
        "https://github.com/poudelef/Datasets/raw/refs/heads/main/"
        "DeepLearning_data_2type_auto-20251122T031133Z-1-001.zip?download="
    )
    dataset_root = load_data(data_url)

    # 2. Preprocess → this returns real tf.data.Dataset objects
    train_ds, test_ds, num_classes = preprocess_data(dataset_root)

 
    print("Detected classes:", class_names(dataset_root / "train"))
    print("Image counts train:", count_images_in_subfolders(dataset_root / "train"))
    print("Image counts test:", count_images_in_subfolders(dataset_root / "test"))

    # 3. Train model
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

    # 5. Reload and evaluate
    loaded = tf.keras.models.load_model(model_path, compile=False)
    print("Evaluating loaded model:")
    print(loaded.evaluate(test_ds))

if __name__ == "__main__":
    main()
