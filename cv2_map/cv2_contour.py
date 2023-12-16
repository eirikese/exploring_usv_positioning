
import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(image_path):
    # Reading image
    img = cv2.imread(image_path)

    # Converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Compute the distance transform
    dist_transform = cv2.distanceTransform(threshold, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)

    # Find contours on the sure foreground
    contours, _ = cv2.findContours(np.uint8(sure_foreground), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))

    # Drawing all contours except the largest one on the original image
    for i, contour in enumerate(contours):
        if i != largest_contour_index:
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)

    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# Paths to your images
image_path1 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\evaluation_data\image_7.png"
image_path2 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\evaluation_data\image_2.png"

# Process both images
img1 = process_image(image_path1)
img2 = process_image(image_path2)

# Use matplotlib to display the images side by side
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.axis('on')
plt.xlabel('pixels')
plt.ylabel('pixels')
plt.title('Gaussian noise = 0.01 [m]')

plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.axis('on')
plt.xlabel('pixels')
plt.ylabel('pixels')
plt.title('Gaussian noise = 0.1 [m]')

plt.show()
