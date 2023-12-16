import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image_and_get_contours(path):
    # Reading the image
    img = cv2.imread(path)

    # Converting the image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Compute the distance transform
    contour_threshold = 0.1
    dist_transform = cv2.distanceTransform(threshold, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(dist_transform, contour_threshold * dist_transform.max(), 255, 0)

    # Find contours on the sure foreground
    contours, _ = cv2.findContours(np.uint8(sure_foreground), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return img, contours

# File paths
path1 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\evaluation_data\image_1.png"
path2 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\template_data\image_0.png"

# Process images and get contours
img1, contours1 = process_image_and_get_contours(path1)
img2, contours2 = process_image_and_get_contours(path2)

# Check for shape similarity and mark dissimilar contours in red
for i, contour1 in enumerate(contours1):
    min_similarity = float('inf')
    for j, contour2 in enumerate(contours2):
        similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
        min_similarity = min(min_similarity, similarity)
    if min_similarity > 0.5:
        cv2.drawContours(img1, [contour1], -1, (0, 0, 255), 2)

for i, contour2 in enumerate(contours2):
    min_similarity = float('inf')
    for j, contour1 in enumerate(contours1):
        similarity = cv2.matchShapes(contour2, contour1, cv2.CONTOURS_MATCH_I1, 0)
        min_similarity = min(min_similarity, similarity)
    if min_similarity > 0.5:
        cv2.drawContours(img2, [contour2], -1, (0, 0, 255), 2)

# Plot the images
plt.figure(figsize=(30,10))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Evaluation Data')

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Template Data')

# Overlay the images for the third subplot as previously done
overlay = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Overlay')

plt.show()
