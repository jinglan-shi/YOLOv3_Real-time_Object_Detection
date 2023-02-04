import cv2

# Prepare Trackbars
# Define empty function
def do_nothing(x):
    pass

# Create a holding window for trackbars
cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

# Create trackbars for different color ranges
cv2.createTrackbar('min_blue', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('min_green', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('min_red', 'Track Bars', 0, 255, do_nothing)

cv2.createTrackbar('max_blue', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('max_green', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('max_red', 'Track Bars', 0, 255, do_nothing)

# Read in image with OpenCV library
image_path = '/Users/jinglanshi/PycharmProjects/YOLOv3 for Github/Familiar with OpenCV/object-to-detect.jpeg'
image_BGR = cv2.imread(image_path)
image_BGR = cv2.resize(image_BGR, (800, 800))

# Show original image
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image_BGR)

# Convert image from BGR space to HSV spave
image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)

# Show HSV image
cv2.namedWindow('HSV Image', cv2.WINDOW_NORMAL)
cv2.imshow('HSV Image', image_HSV)

# Choose right colors for mask
while True:
    min_blue = cv2.getTrackbarPos('min_blue', 'Track Bars')
    min_green = cv2.getTrackbarPos('min_green', 'Track Bars')
    min_red = cv2.getTrackbarPos('min_red', 'Track Bars')

    max_blue = cv2.getTrackbarPos('max_blue', 'Track Bars')
    max_green = cv2.getTrackbarPos('max_green', 'Track Bars')
    max_red = cv2.getTrackbarPos('max_red', 'Track Bars')

    # Implement mask with chosen color ranges
    mask = cv2.inRange(image_HSV, (min_blue, min_green, min_red), (max_blue, max_green, max_red))

    # Show binary image (mask)
    cv2.namedWindow('Binary Image with Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Image with Mask', mask)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all windows
cv2.destroyAllWindows()

# Print final chosen color ranges
print(f"min_blue: {min_blue}, min_green: {min_green}, min_red: {min_red}")
print(f"max_blue: {max_blue}, max_green: {max_green}, max_red: {max_red}")