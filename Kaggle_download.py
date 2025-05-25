import os
import shutil

import kagglehub

# Download latest version
path = kagglehub.dataset_download("deepu1109/star-dataset")

dest_path = "./Data/raw"

for filename in os.listdir(path):
    shutil.move(os.path.join(path, filename), dest_path)

print("Path to dataset files:", path)