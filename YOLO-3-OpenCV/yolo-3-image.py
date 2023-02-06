import random

import numpy as np
import cv2
import time

# Read in image
image_BGR = cv2.imread('images/cat-walking.png')

# Show original image
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image_BGR)
cv2.waitKey(0)
cv2.destroyWindow('Original Image')

# Checkpoint showing image shape
print(f"Image shape: {image_BGR.shape}")

# Get spatial dimension of the input image
h, w = image_BGR.shape[0], image_BGR.shape[1]

#################################################################################
# Get blob from input image

blob = cv2.dnn.blobFromImage(image_BGR, 1/255., (1920, 1280), swapRB=True, crop=False)

# Checkpoint
print(f"Image shape: {image_BGR.shape}")
print(f"Blob shape: {blob.shape}")

# Show blob image
blob_to_show = blob[0].transpose(1,2,0)
print(f"Shape of blob_to_show: {blob_to_show.shape}")
cv2.namedWindow('Blob Image', cv2.WINDOW_NORMAL)
cv2.imshow('Blob Image', cv2.cvtColor(blob_to_show, cv2.COLOR_RGB2BGR))
cv2.waitKeyEx(0)
cv2.destroyWindow('Blob Image')

#################################################################################
# Load YOLO v3 network

# Load coco dataset label names from file
with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]
# Check point
print(f"Label names: {labels}")

# Load trained YOLOv3 network
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg', 'yolo-coco-data/yolov3.weights')

# Get output layers' names
layers_names_output = list(network.getUnconnectedOutLayersNames())
# Check point
print(f"Name of output layers: {layers_names_output}")

# Set thresholds for eliminating waek predictions and weak bounding boxes
probability_minimum = 0.5
threshold = 0.3

# Generate color scale for representing each detected object
colors = np.random.randint(0, 255, (len(labels), 3), dtype='uint8')
# Checkpoint
print(f"Color scale type: {type(colors)}")
print(f"Shape of generated colors: {colors.shape}")
print(f"An example of the colors: {colors[0]}")

#################################################################################
# Implement forward pass

# Take input image
network.setInput(blob)
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()

# Show the time consumption
print(f"The object detection forward pass took {(end-start):.5f} seconds.")

#################################################################################
# Get bounding boxes
bounding_boxes = []
confidences = []
class_indices = []

# Iterate each scale of bounding boxes
for scale_boxes in output_from_network:
    for detected_objects in scale_boxes:
        scores = detected_objects[5:]
        current_class = np.argmax(scores)
        current_confidence = scores[current_class]

        # Eliminate predictions with low confidence
        if current_confidence > probability_minimum:
            current_box = detected_objects[:4] * np.array([w, h, w, h])
            center_x, center_y, box_width, box_height = current_box
            x_min = int(center_x - box_width/2)
            y_min = int(center_y - box_height/2)

            # Add the filtered results to lists
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(current_confidence))
            class_indices.append(current_class)

# Perform Non-Max Suppression to eliminate weak bounding box
results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

# Check point
print(f"Number of bounding boxes before NMS: {len(bounding_boxes)}")
print(f"Number of bounding boxes after NMS: {len(results)}")

# iterate through the resulted bounding boxes
for i in results:
    x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
    box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

    # Get the color for the current detected object/class
    current_box_color = colors[class_indices[i]].tolist()

    # Draw bounding box on the original image
    cv2.rectangle(image_BGR, (x_min, y_min), (x_min+box_width, y_min+box_height), current_box_color, 3)

    # Prepare text to attached on the bounding box
    current_text = f"{labels[class_indices[i]]}: {confidences[i]}"
    cv2.putText(image_BGR, current_text, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 1, current_box_color, 2)

# Show original image with detected objects
cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
cv2.imshow('Detections', image_BGR)
cv2.waitKey(0)
cv2.destroyWindow('Detections')


