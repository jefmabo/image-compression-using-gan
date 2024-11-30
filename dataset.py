from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os
import shutil

def download(kaggle_dataset = 'jessicali9530/celeba-dataset', clear_dataset = True):    
    api = KaggleApi()
    download_path = "dataset/all_images"

    if not os.path.exists(download_path):
        os.makedirs(download_path)
    elif clear_dataset:
        print("Clearing dataset ...")
        clear_dataset_folder(download_path)

    print("Downloading dataset ...")
    api.dataset_download_files(kaggle_dataset, path=download_path)

    print("Unziping dataset ...")
    unzip_dataset(download_path)

    print("Moving dataset files ...")
    move_images_to_root_folder(download_path)

    print("Getting all dataset files path ...")
    return get_all_files_path()

def clear_dataset_folder(path):
    for item_name in os.listdir(path):
        item_path = os.path.join(path, item_name)

        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def unzip_dataset(path):
    zip_files = [file for file in os.listdir(path) if file.endswith('.zip')]
    
    if zip_files:
        with zipfile.ZipFile(os.path.join(path, zip_files[0]), 'r') as zip_ref:
            zip_ref.extractall(path)
    
    for zip_file in zip_files:
        os.remove(os.path.join(path, zip_file))

def move_images_to_root_folder(path):
    for subdir, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
                shutil.move(file_path, os.path.join(path, file))
            else:
                print(f"{file} is not a image. Removing ...")
                os.remove(file_path)
    
    for subdir, dirs, files in os.walk(path):                
        for dir in dirs:
            shutil.rmtree(os.path.join(path, dir))

def get_all_files_path():
    file_paths = []
    try:
        base_path = "dataset/all_images"
        file_paths = [os.path.join(base_path, file) for file in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, file))]
    except:
        print(f"Path '{base_path}' is empty ...")
    return file_paths