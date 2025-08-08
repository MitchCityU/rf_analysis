import os
os.environ["KAGGLE_USERNAME"] = "mitchellfade"
os.environ["KAGGLE_KEY"] = "ENTER API KEY HERE"

import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Download dataset
dataset = "matthewphelps/libriiq-dwingeloo"
download_dir = "libriiq_dwingeloo"
os.makedirs(download_dir, exist_ok=True)

api.dataset_download_files(dataset, path=download_dir, unzip=True)

print(f"Dataset downloaded to: {download_dir}")