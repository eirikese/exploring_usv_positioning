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

def get_equally_spaced_points(contour, num_points=50):
    """Sample a fixed number of equally spaced points from a contour"""
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # If there are not enough points, return the original contour
    if len(approx) < num_points:
        return approx

    idx = np.round(np.linspace(0, len(approx) - 1, num_points)).astype(int)
    return approx[idx]

# File paths
path1 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\evaluation_data\image_0.png"
path2 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\template_data\image_0.png"

# Process images and get contours
img1, contour1 = process_image_and_get_contour(path1)
img2, contour2 = process_image_and_get_contour(path2)

# Matching shapes
ret = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
matching_result = "Shape Similarity: " + str(ret)

# Sample a fixed number of points from both contours
num_points = 50
sampled_points1 = get_equally_spaced_points(contour1, num_points)
sampled_points2 = get_equally_spaced_points(contour2, num_points)

# Find the homography using the sampled points
retval, mask = cv2.findHomography(sampled_points2, sampled_points1)

# Use the found homography to warp the template image
warped_img2 = cv2.warpPerspective(img2, retval, (img1.shape[1], img1.shape[0]))

# Create an overlay by taking the weighted sum of the images
overlay = cv2.addWeighted(img1, 0.7, warped_img2, 0.3, 0)

# Plot side by side
#plt.figure(figsize=(30,8))

plt.subplot(1,3,1)
cv2.drawContours(img1, [contour1], -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis('on')
plt.title('Target')

plt.subplot(1,3,2)
cv2.drawContours(img2, [contour2], -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.axis('on')
plt.title('Source')

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis('on')
plt.title('Overlay')

#plt.suptitle(matching_result)
plt.tight_layout()
plt.show()
