# In this script, we prepare txt files for custom data for feeding in YOLO model.

# Files train.txt and test.txt look like following (every path is in a new line):
# /home/my_name/labelled-images/image001.jpg
# /home/my_name/labelled-images/image002.jpg
# /home/my_name/labelled-images/image003.jpg
# ...
# /home/my_name/labelled-images/image799.jpg
# /home/my_name/labelled-images/image800.jpg


import os

# Set full path of images folder
full_path_to_images = '/Users/jinglanshi/Desktop/YOLOv3 for Github/labelled-data/video'

# Check current working directory
print(f"Current working directory: {os.getcwd()}")

# Go the the directory with all images
os.chdir(full_path_to_images)
print(f"Current working directory: {os.getcwd()}")

# Define an empty list to write image path in
p = []

# Walk through all directories and files to extract target information
for dirpaths, dirnames, filenames in os.walk('.'):
    # Iterate through all images
    for f in filenames:
        # Filter all images
        if f.endswith('.jpeg'):
            # Get image path
            img_p = os.path.join(full_path_to_images, f)
            # Append the path to list
            p.append(img_p + '\n')

# Split into train set and test set
size = int(len(p)*0.15)
p_train = p[:-size]
p_test = p[-size:]

# Create train.txt and test.txt and write information in
with open('train.txt', 'w') as train_txt:
    for term in p_train:
        train_txt.write(term)

with open('test.txt', 'w') as test_txt:
    for term in p_test:
        test_txt.write(term)
