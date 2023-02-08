# Convert annotations written in csv file into YOLO format

import os
import pandas as pd

full_path_to_csv = '/Users/jinglanshi/OIDv4_ToolKit/OID/csv_folder'
full_path_to_images = '/Users/jinglanshi/OIDv4_ToolKit/OID/Dataset/train/Bird_Dog_Person'

##########################################################################################
# Extract LabelName from annotation file corresponding to labels of images we have       #
##########################################################################################

# Label list containing the image categories of interest
labels = ['Bird', 'Dog', 'Person']

# Read in class-description file
class_info = pd.read_csv(full_path_to_csv+'/'+'class-descriptions-boxable.csv', usecols=[0, 1], header=None)

# Create empty list for accommodating encrypted labels
encrypted_labels = []

# Iterate through all labels
for l in labels:
    # Match labels values
    sub_classes = class_info.loc[class_info[1] == l]
    # Get LabelName
    ln = sub_classes.iloc[0][0]
    # Append the entrypted name to empty list
    encrypted_labels.append(ln)


##########################################################################################
# Extract information and preprocess into YOLO format                                    #
##########################################################################################

# Read in annotation file
annotation = pd.read_csv(full_path_to_csv+'/'+'train-annotations-bbox.csv',
                         usecols=['ImageID',
                                  'LabelName',
                                  'XMin',
                                  'XMax',
                                  'YMin',
                                  'YMax'])

# Retrieve images that is of categories of interest
# Make sure initial DataFrame will not be modified by using .copy()
sub_annotation = annotation.loc[annotation['LabelName'].isin(encrypted_labels)].copy()

# Add required information for YOLO format to retrieved DataFrame
sub_annotation['Class Number'] = ''
sub_annotation['center x'] = ''
sub_annotation['center y'] = ''
sub_annotation['width'] = ''
sub_annotation['height'] = ''

# Get indices of classes
for i in range(len(encrypted_labels)):
    sub_annotation.loc[sub_annotation['LabelName'] == encrypted_labels[i], 'Class Number'] = i

# Calculate parameters information
sub_annotation['center x'] = (sub_annotation['XMax']+sub_annotation['XMin'])/2
sub_annotation['center y'] = (sub_annotation['YMax']+sub_annotation['YMin'])/2
sub_annotation['width'] = sub_annotation['XMax']-sub_annotation['XMin']
sub_annotation['height'] = sub_annotation['YMax']-sub_annotation['YMin']

# Remove unnecessary informations
annotation_to_write = sub_annotation.loc[:, ['ImageID',
                                             'Class Number',
                                             'center x',
                                             'center y',
                                             'width',
                                             'height']].copy()


##########################################################################################
# Write annotation information into the directory where images are                       #
##########################################################################################

# Change to images directory
os.chdir(full_path_to_images)

for dirpath, dirnames, filenames in os.walk(full_path_to_images):
    # Go through all images
    for f in filenames:
        # Check for proper image format
        if f.endswith('jpg'):
            # Get only image ID
            img_name = f[:-4]
            # Pair up
            img_info = annotation_to_write.loc[annotation_to_write['ImageID'] == img_name]
            # Get only needed information
            img_annotation = img_info.loc[:, ['Class Number',
                                              'center x',
                                              'center y',
                                              'width',
                                              'height']].copy()

            # Write into txt file
            write_in_path = full_path_to_images+'/'+img_name+'.txt'
            img_annotation.to_csv(write_in_path, header=False, index=False, sep=' ')

