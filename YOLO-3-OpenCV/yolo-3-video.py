import numpy as np
import time
import cv2


###########################################################################
# Read in video
video = cv2.VideoCapture('videos/Bangkok-traffic.mp4')

# Prepare variables for writing video frames
writer = None
# Prepare variables for spacial dimensions of frames
h, w = None, None


###########################################################################
# Load YOLOv3 network

# Get all classes labels of coco dataset
with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

# Load trained YOLOv3 object detector
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg', 'yolo-coco-data/yolov3.weights')

# Get the names of the output layers
output_layers_names = list(network.getUnconnectedOutLayersNames())

# Set thresholds to eliminate weak predictions and weak bounding boxes
minimum_probability = 0.5
threshold = 0.3

# Generate colors for each object bounding box
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


###########################################################################
# Read in video frames

# Define variabls for counting frames and processing time
f = 0
t = 0

# Define loop for catching frames
while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        break
    if h is None or w is None:
        h, w = frame.shape[:2]

    # Process read frame to get blob
    blob = cv2.dnn.blobFromImage(frame, 1/255., (416, 416), swapRB=True, crop=False)

    # Implement forward pass through only the output layers and count frames and time
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(output_layers_names)
    end = time.time()

    f += 1
    t += end-start
    print(f"Frame {f} took {end-start} seconds.")


    # Get bounding boxes
    bounding_boxes = []
    confidences = []
    class_indices = []

    # Go through different scale of bounding boxes
    for scaled_boxes in output_from_network:
        # Go through all detected objects
        for detected_objects in scaled_boxes:
            # Get 80 class probabilities for current detected object
            scores = detected_objects[5:]
            # Get index of class with the maximum probability
            current_class = np.argmax(scores)
            # Get the value of defined class
            current_confidence = scores[current_class]

            # Eliminate predictions with probability lower then minimum probability
            if current_confidence > minimum_probability:
                current_box = detected_objects[:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = current_box
                x_min = int(x_center - box_width/2)
                y_min = int(y_center - box_height/2)

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(current_confidence)
                class_indices.append(current_class)

    # Perform Non-max suppression to eliminate weak bounding boxes
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, minimum_probability, threshold)

    # Draw bounding boxed
    if len(results)>0:
        for i in results:
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            current_box_color = colors[class_indices[i]].tolist()
            cv2.rectangle(frame, (x_min, y_min), (x_min+box_width, y_min+box_height), current_box_color, 2)
            current_box_label = f"{labels[int(class_indices[i])]}: {(confidences[i]):.4f}"
            # Put label on bounding box
            cv2.putText(frame, current_box_label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_box_color, 1)


    ###########################################################################
    # Write processed frame into file

    # Initialize writer
    if writer is None:
        # Construct code of the codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('videos/traffic-cars-result.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
        # Note that the write-in size needs to match the read-in size from video.read(), otherwise, the file writing would fail.

    # Write processed frame
    writer.write(frame)


# Print final results
print()
print(f"Total number of frames: {f}")
print(f"Total process time: {t:.5f} seconds.")
print(f"FPS: {round((f/t), 1)}")

# Release video reader and writer
video.release()
writer.release()