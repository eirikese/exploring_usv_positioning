import cv2
import numpy as np
from matplotlib import pyplot as plt

# File paths
path1 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\evaluation_data\image_4.png"
path2 = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\template_data\image_0.png"

def find_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0  # Arbitrary center if contour is too small
    return (cX, cY) 

def process_image_and_get_contour(path):
    # Reading the image
    img = cv2.imread(path)
    # Converting the image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Setting threshold of gray image
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

def find_longest_side_angle(contour):
    # Find the convex hull of the contour
    hull = cv2.convexHull(contour)
    # Calculate pairwise distances and find the two farthest points
    dists = np.linalg.norm(hull - hull[:, np.newaxis], axis=-1)
    farthest_pair = np.unravel_index(dists.argmax(), dists.shape)
    # Find the two farthest points
    point1 = tuple(hull[farthest_pair[0]][0])
    point2 = tuple(hull[farthest_pair[1]][0])
    # Calculate the angle of the line formed by these two points
    angle = np.degrees(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
    return angle

def overlay_images(img1, img2, contour2):
    # Convert the template image to green except for the white background
    green_template = np.zeros_like(img2)
    green_mask = img2.mean(axis=2) < 250  # Mask of non-white pixels
    green_template[green_mask] = [0, 255, 0]  # Assign green color
    # Overlay the images
    overlay = cv2.addWeighted(img1, 0.5, green_template, 0.5, 0)
    return overlay

# Process images and get contours
img1, contour1 = process_image_and_get_contour(path1)
img2, contour2 = process_image_and_get_contour(path2)

# Calculate the angles of the longest sides
angle1 = find_longest_side_angle(contour1)
angle2 = find_longest_side_angle(contour2)

# Calculate the centers of the contours
center1 = find_contour_center(contour1)
center2 = find_contour_center(contour2)

# Calculate the translation from center1 to center2
translation_vector = (center2[0] - center1[0], center2[1] - center1[1])

# Calculate the angle to rotate the template image
rotation_angle = angle2 - angle1

# Rotate the template image
M = cv2.getRotationMatrix2D((img2.shape[1] / 2, img2.shape[0] / 2), rotation_angle, 1)
rotated_img2 = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))

# Create the overlay image with the rotated template colored green
overlay = overlay_images(img1, rotated_img2, contour2)

# Plot the images
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
cv2.drawContours(img1, [contour1], -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Target, rotated 10 degrees')

plt.subplot(1, 3, 2)
cv2.drawContours(img2, [contour2], -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Source')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f'Source rotated: {rotation_angle:.2f} degrees to fit Target')

# Add plot description with rotation and translation details
plot_description = f"The template has been rotated by {rotation_angle:.2f} degrees and "\
                   f"translated by ({translation_vector[0]}, {translation_vector[1]}) pixels "\
                   f"to align with the sample."
#plt.figtext(0.5, 0.02, plot_description, ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.tight_layout()
plt.show()
