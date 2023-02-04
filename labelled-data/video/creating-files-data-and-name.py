# In addition to 'train.txt' and 'test.txt' files,
# we need 'labelled_data.data' and 'classes.names' files,
# wherein, the 'labelled_data.data' file contains the following:
# 1. classes = 2
# 2. train = /home/my_name/train.txt
# 3. valid = /home/my_name/test.txt
# 4. names = /home/my_name/classes.names
# 5. backup = backup


import os

# Define working directory
full_path_to_images = '/Users/jinglanshi/Desktop/YOLOv3 for Github/labelled-data/video'
print(os.getcwd())

# Prepare classes.names file

# Define loop for counting classes
c = 0

# Create 'classes.names' file by copying from 'classes.txt'
with open('classes.txt', 'r') as txt, \
     open('classes.names', 'w') as names:
    # Go through all lines in txt file
    for term in txt:
        # Copy it into names file
        names.write(term)

        # Increment counter
        c += 1


# Prepare labelled-data.data file

with open('labelled_data.data', 'w') as data:
    # 1. classes = 2
    data.write(f"classes = {c}\n")
    # 2. train = /home/my_name/train.txt
    data.write(f"train = {os.path.join(full_path_to_images, 'train.txt')}\n")
    # 3. valid = /home/my_name/test.txt
    data.write(f"valid = {os.path.join(full_path_to_images, 'test.txt')}\n")
    # 4. names = /home/my_name/classes.names
    data.write(f"names = {os.path.join(full_path_to_images, 'classes.names')}\n")
    # 5. backup = backup
    data.write('backup = backup')