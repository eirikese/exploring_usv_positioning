import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image_and_get_contour(path):
    # Reading the image
    img = cv2.imread(path)
    
    # Converting image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Compute the distance transform
    contour_threshold = 0.05
    dist_transform = cv2.distanceTransform(threshold, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(dist_transform, contour_threshold * dist_transform.max(), 255, 0)

    # Find contours
    contours, _ = cv2.findContours(np.uint8(sure_foreground), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sorting the contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Assuming the second largest contour in the image is the shape of interest
    contour = sorted_contours[1]

    return img, contour

def rotate_image_based_on_contour(img, contour):
    # Get the bounding rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    
    # Get the longest side of the bounding rectangle
    side_lengths = [np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])]
    long_side_index = np.argmax(side_lengths)
    
    # Get the two points of the longest side
    pt1 = box[long_side_index]
    pt2 = box[(long_side_index+1) % 4]
    
    # Compute the angle based on the longest side
    angle = np.degrees(np.arctan2(pt2[1]-pt1[1], pt2[0]-pt1[0]))
    
    # Rotate the image
    rotation_matrix = cv2.getRotationMatrix2D(tuple(rect[0]), angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

    return rotated_img

# File paths
path1 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\evaluation_data\image_1.png"
path2 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\template_data\image_0.png"

# Process images and get contours
img1, contour1 = process_image_and_get_contour(path1)
img2, contour2 = process_image_and_get_contour(path2)

# Rotate the images based on their longest side
rotated_img1 = rotate_image_based_on_contour(img1, contour1)
rotated_img2 = rotate_image_based_on_contour(img2, contour2)

# Plot the images
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(rotated_img1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Rotated Evaluation Data')

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(rotated_img2, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Rotated Template Data')

plt.show()
