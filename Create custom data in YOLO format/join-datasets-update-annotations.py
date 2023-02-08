# This project uses two datasets of different sources:
# 1. One is download from Open Images Dataset;
# 2. The other is the one shoot in real-life scenarios.

# This script is to join datasets of two sources into one directory
# and create required files for training in Darknet framework.


import os

# Define paths to datasets to be joint and to target dataset
full_path_to_downloaded_data = '/Users/jinglanshi/OIDv4_ToolKit/OID/Dataset/train/Bird_Dog_Person'
full_path_to_custom_data = '/Users/jinglanshi/Desktop/YOLOv3 for Github/Create custom data in YOLO format/custom-data'
full_path_to_joint_data = '/Users/jinglanshi/Desktop/YOLOv3 for Github/Create custom data in YOLO format/joint-dataset'




##############################################################################################################
# Create train.txt and text.txt files containing paths of all images                                         #
##############################################################################################################

# Change to target directory
os.chdir(full_path_to_joint_data)


# Define an empty list accommodating images paths
p = []


# Do through all directories
for dirpath, dirnames, filenames in os.walk(full_path_to_joint_data):
    # Iterate all files in directory
    for f in filenames:
        # Filter only images
        if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('png'):
            # Get path of the current image
            img_path = os.path.join(dirpath, f)
            # Append to the empty list line by line
            p.append(img_path+'\n')


# Split into train set and test set
split_size = int(len(p) * 0.15)
train_p = p[:-split_size]
test_p = p[-split_size:]


# Write into separate .txt files
with open('train.txt', 'w') as train_txt:
    # Go through all elements of train list
    for e in train_p:
        train_txt.write(e)

with open('test.txt', 'w') as test_txt:
    for e in test_p:
        test_txt.write(e)




##############################################################################################################
# Create classes.names file                                                                                  #
##############################################################################################################

# Define lists for classes from custom and downloaded data
custom_classes = []
downloaded_classes = []

# Get all the unique class names for two datasets
joined_classes = set()


# Read classes from classes.names files and fill the empty lists
with open(os.path.join(full_path_to_custom_data, 'classes.names'), 'r') as custom_names, \
     open(os.path.join(full_path_to_downloaded_data, 'classes.names'), 'r') as downloaded_names:
    # Go through the file line by line
    for line in custom_names:
        # Fill the list and set with lower-cased class name
        custom_classes.append(line.lower().strip())
        joined_classes.add(line.lower().strip())

    for line in downloaded_names:
        downloaded_classes.append(line.lower().strip())
        joined_classes.add(line.lower().strip())

# Convert the set into list of only unique class names
joined_classes = list(joined_classes)


# Create classes.names for joined dataset
with open(os.path.join(full_path_to_joint_data, 'classes.names'), 'w') as names:
    # Go through all elements of the joined classes list
    for e in joined_classes:
        # Write into file line by line
        names.write(e+'\n')




##############################################################################################################
# Create joined_data.data file for joined dataset                                                            #
##############################################################################################################
# The file contains the following information:
# classes = n (integer)
# train = /home/my_name/train.txt
# valid = /home/my_name/test.txt
# names = /home/my_name/classes.names
# backup = backup

with open(os.path.join(full_path_to_joint_data, 'joined_data.data'), 'w') as data:
    # Write number of classes
    data.write(f"classes = {str(len(joined_classes))}\n")

    # Write the location of train.txt file
    data.write(f"train = {os.path.join(full_path_to_joint_data, 'train.txt')}\n")

    # Write the location of test.txt file
    data.write(f"valid = {os.path.join(full_path_to_joint_data, 'test.txt')}\n")

    # Write the location of classes.names file
    data.write(f"names = {os.path.join(full_path_to_joint_data, 'classes.names')}\n")

    # Location where to save weights
    data.write(f"backup = backup")




##############################################################################################################
# Update separate annotation information and write into joined da                                            #
##############################################################################################################

# 1. First update and write custom data's annotations

# Change to custom dataset directory
os.chdir(full_path_to_custom_data)

# Go through all directories
for dirpath, dirnames, filenames in os.walk('.'):
    # Go through all files
    for f in filenames:
        # Get proper file name without extension
        if f.endswith('jpeg') or f.endswith('jpg') or f.endswith('png'):
            txt_name = os.path.splitext(f)[0]

            # Get path to annotation file of current image
            custom_txt_path = os.path.join(full_path_to_custom_data, f"{txt_name}.txt")

            # Define the path to write the annotation to
            joined_txt_path = os.path.join(full_path_to_joint_data, f"{txt_name}.txt")

            # Copy annotation of current image from custom dataset directory
            # write into joined dataset directory
            # update class number in the annotation file
            with open(custom_txt_path, 'r') as custom,\
                 open(joined_txt_path, 'w') as joined:
                 # Go through all lines in the file
                 for line in custom:
                     # Get current class number before joining together
                     c_number = int(line[:1])

                     # Get the class name in custom data context
                     c_name = custom_classes[c_number]

                     # Get updated class number
                     updated_c_number = joined_classes.index(c_name)

                     # Update the new class number to annotation information
                     updated_line = f"{updated_c_number}{line[1:]}"

                     # Write into joined data
                     joined.write(updated_line)




# 2. Then update and write downloaded data's annotations

# Change to downloaded dataset directory
os.chdir(full_path_to_downloaded_data)

# Go through all directories
for dirpath, dirnames, filenames in os.walk('.'):
    # Go through all files
    for f in filenames:
        # Get proper file name without extension
        if f.endswith('jpeg') or f.endswith('jpg') or f.endswith('png'):
            txt_name = os.path.splitext(f)[0]

            # Get path to annotation file of current image
            downloaded_txt_path = os.path.join(full_path_to_downloaded_data, f"{txt_name}.txt")

            # Define the path to write the annotation to
            joined_txt_path = os.path.join(full_path_to_joint_data, f"{txt_name}.txt")

            # Copy annotation of current image from downloaded dataset directory
            # write into joined dataset directory
            # update class number in the annotation file
            with open(downloaded_txt_path, 'r') as downloaded, \
                    open(joined_txt_path, 'w') as joined:
                # Go through all lines in the file
                for line in downloaded:
                    # Get current class number before joining together
                    c_number = int(line[:1])

                    # Get the class name in downloaded data context
                    c_name = downloaded_classes[c_number]

                    # Get updated class number
                    updated_c_number = joined_classes.index(c_name)

                    # Update the new class number to annotation information
                    updated_line = f"{updated_c_number}{line[1:]}"

                    # Write into joined data
                    joined.write(updated_line)