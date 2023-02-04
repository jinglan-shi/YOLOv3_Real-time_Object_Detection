import cv2
import numpy as np
import time

#########################################################################
# Prepare camera                                                        #
#########################################################################

# Define 'VideoCapture' object to read in real-time video
camera = cv2.VideoCapture(0)

# Define variables for spatial dimension of video
h, w = None, None



#########################################################################
# Load YOLO v3 network                                                  #
#########################################################################

# Prepare coco dataset's classes names
with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

# Load trained YOLO v3 objects detector
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')

# Get only the output layers names
output_layers_names = list(network.getUnconnectedOutLayersNames())

# Set thresholds for filtering bounding boxes
minimum_probability = 0.5
threshold = 0.3

# Generate colors for bounding box of each detected class
colors = np.random.randint(0, 255, (len(labels), 3), 'uint8')



#########################################################################
# Define loop to read in streaming video                                #
#########################################################################

while True:
    # Capture frame-by-frame from camera
    ret, frame = camera.read()

    # Get spatial dimension for all frames once
    if h is None or w is None:
        h, w = frame.shape[:2]



    #########################################################################
    # Inference stage                                                       #
    #########################################################################

    # Preprocess current frame to get blob for the network's use
    blob = cv2.dnn.blobFromImage(frame, 1/255., (416, 416), swapRB=True, crop=False)

    # Forward pass through only output layers and calcure processing time
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(output_layers_names)
    end = time.time()

    # Show the spent time for single current frame
    print(f"Current frame took {end-start} seconds.")


    # Get all bounding boxes and associated parameters
    bounding_boxes = []
    confidences = []
    class_indices = []

    # Iterate throught bounding boxes of different scales
    for results in output_from_network:
        # Go through every single detection
        for detected_object in results:
            # Get all 80 classes' scores(probabilities)
            scores = detected_object[5:]
            # Get the current predicted class
            current_class = np.argmax(scores)
            # Get the current class confidence
            current_confidence = scores[current_class]


            # Eliminate predictions with weak confidence level
            if current_confidence > minimum_probability:
                # Scale the coordinates of the bounding box back to initial frame size
                # Note: output from YOLOv3 takes the order of [Pc, bounding_box_params, class_probabilities]
                current_box = detected_object[:4] * np.array([w, h, w, h])

                # Get top-left coordinates
                x_center, y_center, box_width, box_height = current_box
                x_min = int(x_center - box_width/2)
                y_min = int(y_center - box_height/2)

                # Add the filtered boxes into the list
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(current_confidence)
                class_indices.append(current_class)


    # Perform Non-max suppression
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, minimum_probability, threshold)



    #########################################################################
    # Display results                                                       #
    #########################################################################

    # Draw bounding box on current frame
    if len(results) > 0:
        for i in results:
            # Get coordinates of the bounding box
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Draw bounding box
            current_color = colors[class_indices[i]].tolist()
            cv2.rectangle(frame, (x_min, y_min), (x_min+box_width, y_min+box_height),
                          current_color, 2)

            # Prepare text to be attached on current bounding box
            current_text = f"{labels[class_indices[i]]}: {(confidences[i]):.4f}"

            # Attach text to the bounding box
            cv2.putText(frame, current_text, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 1)

    # Show results in real-time video
    cv2.namedWindow('YOLO v3 Real Time Detector', cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO v3 Real Time Detector', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



#########################################################################
# Release camera and kill all windows                                   #
#########################################################################
camera.release()
cv2.destroyAllWindows()