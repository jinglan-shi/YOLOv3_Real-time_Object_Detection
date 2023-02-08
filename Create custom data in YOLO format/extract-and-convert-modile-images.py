# Custom data from real-life scenario are taken by IPhone, some of the images are in .heic format
# These images need to be converted into proper format for Darknet framework
# In this script, images of different formats will be extracted for converted
# And at last, they are stored in one directory.

import os
import re
import shutil
import zipfile
from PIL import Image
from pillow_heif import register_heif_opener



# Define the directories that contain initial images and processes images
target_dir = '/Users/jinglanshi/Desktop/YOLOv3 for Github/Create custom data in YOLO format/custom-data'
source_dir = '/Users/jinglanshi/Desktop/downloaded_imgs'



# Define a function to extract only .heic image
def extract_heic(f_path, target_dir):
    with zipfile.ZipFile(f_path) as zip_ref:
        members = zip_ref.namelist()
        for m in members:
            # Check for .heic file and extract to target directory
            if m.endswith('.heic'):
                zip_ref.extract(m, target_dir)
    return None



# Extract images and store to target store
for pathdir, dirnames, filenames in os.walk(source_dir):
    # Go through all subdirectories
    for dir in dirnames:
        c = 0
        # Get subdirectory path
        dir_path = os.path.join(pathdir, dir)
        # Iterate all files in current subdirectory
        for f in os.listdir(dir_path):
            # Check for file format
            if f == '.DS_Store':
                continue

            else:
                c += 1
                # Get full file path
                f_path = os.path.join(dir_path, f)
                # Filter .zip file to unzip
                if f_path.endswith('.zip'):
                    extract_heic(f_path, target_dir)
                # All non-zip file moved to target directory
                else:
                    shutil.move(f_path, target_dir)

        print(f"Directory {dir} contains {c} files.")
        print()



# Conver all .heic files into .jpg file and count all files
c = 0
for file in os.listdir(target_dir):
    c += 1
    # Filter all .heic files and extract information
    if file.endswith('heic'):
        register_heif_opener()
        img_path = os.path.join(target_dir, file)
        img = Image.open(img_path)
        # Save into .jpg format
        save_path = re.sub(r'(?i)\.(heic)', '', img_path) + '.jpg'
        img.save(save_path)
        # Delete original .heic file
        os.remove(img_path)
print(f"There are {c} images in total after process.")

