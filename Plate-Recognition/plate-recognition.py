import cv2
from matplotlib import pyplot as plt
import numpy as np
import easyocr
import imutils


# Read the image
img = cv2.imread("car2.png")  # replace with your image path
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter for noise reduction while preserving edges
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
plt.imshow(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB))
plt.title('Processed Image')
plt.show()

# Perform edge detection using Canny
edged = cv2.Canny(bfilter, 30, 200)
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.title('Edge Detection')
plt.show()


# Find contours in the edged image
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)

# Sort contours based on their area and keep the top 10
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Loop over the contours to find the best possible approximation
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:  # Number plate is a rectangle
        location = approx
        break

print("Location: ", location)


# Create a mask with the same dimensions as the original grayscale image
mask = np.zeros(gray.shape, np.uint8)

# Draw the contour of the detected number plate on the mask
new_image = cv2.drawContours(mask, [location], 0, 255, -1)

# Perform bitwise AND to isolate the number plate region
new_image = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title('Masked Image')
plt.show()

# Crop the number plate from the image
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]


plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title('Cropped Number Plate')
plt.show()

# Create an EasyOCR reader object
reader = easyocr.Reader(['en'])

# Read the text from the cropped image
result = reader.readtext(cropped_image)

# Display the extracted text
print("Detected Text:", result)

# Extract the detected text from the OCR result
text = result[0][-2]  # The second-to-last value contains the recognized text

print("Plate:", text)

# Define font and position for the text
font = cv2.FONT_HERSHEY_SIMPLEX

# Draw the text on the original image
res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1]+60),
                  fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

# Draw a rectangle around the detected number plate
res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

# Show the final result
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title('Final Result with Detected Text')
plt.show()
