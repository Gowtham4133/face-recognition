import cv2

image_path = "D:/detect the face /dataset /with_mask /image.jpg"  # Make sure this path is correct
image = cv2.imread(image_path)  # Read the image

# Check if the image is loaded
if image is None:
    print("Error: Image not found or unable to read.")
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    cv2.imshow("Grayscale Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
