from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os

def download(kaggle_dataset = 'jessicali9530/celeba-dataset', clear_dataset = True):    
    api = KaggleApi()
    download_path = "datasets"

    if not os.path.exists(download_path):
        os.makedirs(download_path)
    elif clear_dataset:
        clear_dataset_folder(download_path)

    api.dataset_download_files(kaggle_dataset, path=download_path)

    unzip_dataset(download_path)


def clear_dataset_folder(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        os.remove(file_path)

def unzip_dataset(path):
    zip_files = [file for file in os.listdir(path) if file.endswith('.zip')]
    
    if zip_files:
        with zipfile.ZipFile(os.path.join(path, zip_files[0]), 'r') as zip_ref:
            zip_ref.extractall(path)
    
    for zip_file in zip_files:
        os.remove(os.path.join(path, zip_file))
