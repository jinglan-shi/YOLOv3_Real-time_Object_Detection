import cv2

# Define bounds of founded mask
min_blue, min_green, min_red = 5, 220, 113
max_blue, max_green, max_red = 10, 255, 255

# Define object for reading video from (default) camera
camera = cv2.VideoCapture(0)

# Catch frames from video
while True:
    # Read in frame-by-frame
    _, frame_BGR = camera.read()

    # Convert from BGR to HSV
    frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)

    # Implement mask with found color bounds
    mask = cv2.inRange(frame_HSV,
                       (min_blue, min_green, min_red),
                       (max_blue, max_green, max_red))

    # Show the binary image(mask)
    cv2.namedWindow('Binary frame with mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary frame with mask', mask)

    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Find the biggest contour based on area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Extract the coordinates of the biggest contour if any was found
    if contours:
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])

        # Draw bounding box on current BGR frame
        cv2.rectangle(frame_BGR,
                      (x_min-15, y_min-15),
                      (x_min+box_width+15, y_min+box_height+15),
                      (0, 255, 0),
                      3)

        # Write label text on the bounding box
        label = 'Detected Object'
        cv2.putText(frame_BGR, label,
                    (x_min-5, y_min-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2)

    # Show current BGR frame with detected object
    cv2.namedWindow('Detected object', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected object', frame_BGR)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroying all opened windows
cv2.destroyAllWindows()
