# data_ingestion.py
import requests
import zipfile
import os
from pathlib import Path


def load_data(url: str | None = None) -> Path:
    """
    - If dataset already exists under project_root/data/DeepLearning_data_2type_auto,
      just return that path.
    - Otherwise, try to download ZIP from `url`, extract it, and return the dataset root.
    """
    
    # parents[2] -> End-To-End-Pipeline-
    project_root = Path(__file__).resolve().parents[2]
    extract_path = project_root / "data"
    dataset_root = extract_path / "DeepLearning_data_2type_auto"

    # 1) If already extracted, reuse it
    if dataset_root.exists():
        print(f"Found existing dataset at: {dataset_root}")
        return dataset_root

    if url is None:
        raise RuntimeError("Dataset not found locally and no URL provided to download it.")

    # 2) Otherwise, download
    try:
        print("Downloading dataset from:", url)

        zip_path = project_root / "dataset.zip"

        # Streamed download is safer for large files
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"Extracting to: {extract_path}")
        extract_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_path)

        # Clean up zip
        os.remove(zip_path)

        # If your zip contains a top-level folder with this name, this will work:
        if not dataset_root.exists():
            # fallback: just pick the first subfolder
            candidates = [p for p in extract_path.iterdir() if p.is_dir()]
            if not candidates:
                raise RuntimeError("No folders found after extraction.")
            dataset_root = candidates[0]

        print(f"Dataset ready at: {dataset_root}")
        return dataset_root

    except requests.exceptions.RequestException as e:
        print("Error in load_data (download failed):", e)
        raise
