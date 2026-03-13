import os
import subprocess

DATASET = "mlg-ulb/creditcardfraud"
DATA_DIR = "data"


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading dataset from Kaggle...")

    subprocess.run([
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        DATA_DIR,
        "--unzip"
    ], check=True)

    print("Dataset downloaded successfully!")
    print(f"Location: {DATA_DIR}/creditcard.csv")


if __name__ == "__main__":
    download_dataset()