import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Define the target directory
target_dir = os.path.join(os.getcwd(), 'data', 'raw')

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset
dataset = 'saharnazyaghoobpoor/air-france-reviews-dataset'
api.dataset_download_files(dataset, path=target_dir, unzip=True)

print("Path to dataset files:", target_dir) 