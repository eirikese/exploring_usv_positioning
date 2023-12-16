import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image_and_get_contour(path):
    # reading image
    img = cv2.imread(path)

    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Compute the distance transform
    contour_threshold = 0.05
    dist_transform = cv2.distanceTransform(threshold, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(dist_transform, contour_threshold * dist_transform.max(), 255, 0)

    # Find contours on the sure foreground
    contours, _ = cv2.findContours(np.uint8(sure_foreground), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting the contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Assuming the second largest contour in the image is the shape of interest
    contour = sorted_contours[1]

    return img, contour

def translate_image_to_match_contours(img_to_translate, contour1, contour2):
    # Calculate the centroids of the contours
    M1 = cv2.moments(contour1)
    M2 = cv2.moments(contour2)
    cx1 = int(M1['m10'] / M1['m00'])
    cy1 = int(M1['m01'] / M1['m00'])
    cx2 = int(M2['m10'] / M2['m00'])
    cy2 = int(M2['m01'] / M2['m00'])

    # Compute the translation based on centroids
    dx = cx1 - cx2
    dy = cy1 - cy2

    # Translate the image
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    translated_img = cv2.warpAffine(img_to_translate, translation_matrix, (img_to_translate.shape[1], img_to_translate.shape[0]))

    return translated_img

# File paths
path1 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\evaluation_data\image_3.png"
path2 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\template_data\image_0.png"

# Process images and get contours
img1, contour1 = process_image_and_get_contour(path1)
img2, contour2 = process_image_and_get_contour(path2)

# Translate the template image based on centroids
translated_img2 = translate_image_to_match_contours(img2, contour1, contour2)

# Overlay the images
overlay = cv2.addWeighted(img1, 0.7, translated_img2, 0.3, 0)

# Plot the images
plt.figure(figsize=(30,10))

plt.subplot(1,3,1)
cv2.drawContours(img1, [contour1], -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Evaluation Data')

plt.subplot(1,3,2)
cv2.drawContours(translated_img2, [contour2], -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(translated_img2, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Translated Template Data')

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Overlay')

plt.show()
