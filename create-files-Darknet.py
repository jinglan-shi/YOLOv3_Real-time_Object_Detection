# There are few files needed for Darknet framework:
# • custom_data.data
# • classes.names
# • train.txt
# • test.txt

# Five lines inside custom_data.data are:
# • classes = 3
# • train = /home/my_name/train.txt
# • valid = /home/my_name/test.txt
# • names = /home/my_name/classes.names
# • backup = backup

import os
import random

# Define the working directory
# Note if you have datasets from different sources, this directory varies depends on your case
full_path_to_images = '/Users/jinglanshi/Desktop/YOLOv3_for_Github/Create_custom_data_in_YOLO_format/custom-data'

# Change to target directory
os.chdir(full_path_to_images)

#########################################################################################
# Write train.txt and test.txt files                                                    #
#########################################################################################

# Define an empty list holding paths
p = []

# Walk through all directories
for dirpath, firnames, filenames in os.walk(full_path_to_images):
    # Walk through all images
    for f in filenames:
        # Check for jpg files
        if f.endswith('.jpg'):
            # Get full path of current image
            path_to_write = os.path.join(full_path_to_images, f)
            # Append the matched path to list
            p.append(path_to_write+'\n')

# Split the image into train set and test set
test_size = int(len(p) * 0.15)
# Shuffle the images to disrupt possible pattern
random.seed(42)
random.shuffle(p)
train_p = p[:-test_size]
test_p = p[-test_size:]

# Write image paths into txt file
with open('train.txt', 'w') as train_txt:
    for item in train_p:
        train_txt.write(item)

with open('test.txt', 'w') as test_txt:
    for item in test_p:
        test_txt.write(item)

print(os.getcwd())


#########################################################################################
# Prepare custom_data.data file                                                         #
#########################################################################################

# Define a counter for counting classes numbers
c = 0

# Open classes.txt file and write information into .name file
with open('classes.txt', 'r') as txt,\
     open('classes.names', 'w') as names:

    for line in txt:
        names.write(line)

        # Increment c
        c += 1

# Write required information into .data file
with open('custom_data.data', 'w') as data:
    # classes = 3
    data.write(f"classes = {c}\n")
    # train = /home/my_name/train.txt
    data.write(f"train = {full_path_to_images}/train.txt\n")
    # valid = /home/my_name/test.txt
    data.write(f"valid = {full_path_to_images}/test.txt\n")
    # names = /home/my_name/classes.names
    data.write(f"names = {full_path_to_images}/classes.names\n")
    # backup = backup
    data.write(f"backup = backup")