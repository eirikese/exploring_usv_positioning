import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Constants
template_path = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\template_data\image_0.png"
evaluation_dir = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\evaluation_data"
num_images = 200
noise_levels = np.linspace(0, 0.1, num_images)  # Noise increasing from 0 to 0.5
translation_errors = []
rotation_errors = []

def find_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0  # Arbitrary center if contour is too small
    return (cX, cY) 

def process_image_and_get_contour(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(threshold, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)
    contours, _ = cv2.findContours(np.uint8(sure_foreground), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour = sorted_contours[1]
    return img, contour

def find_longest_side_angle(contour):
    hull = cv2.convexHull(contour)
    dists = np.linalg.norm(hull - hull[:, np.newaxis], axis=-1)
    farthest_pair = np.unravel_index(dists.argmax(), dists.shape)
    point1 = tuple(hull[farthest_pair[0]][0])
    point2 = tuple(hull[farthest_pair[1]][0])
    angle = np.degrees(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
    return angle

# Load the template image and its contour
template_img, template_contour = process_image_and_get_contour(template_path)
template_center = find_contour_center(template_contour)
template_angle = find_longest_side_angle(template_contour)

for i in range(num_images):
    eval_path = os.path.join(evaluation_dir, f'image_{i}.png')
    eval_img, eval_contour = process_image_and_get_contour(eval_path)

    eval_center = find_contour_center(eval_contour)
    eval_angle = find_longest_side_angle(eval_contour)

    translation_error = np.linalg.norm(np.array(eval_center) - np.array(template_center)) - 3
    rotation_error = abs(eval_angle - template_angle) - 10
    if rotation_error > 180:
        rotation_error = 360 - rotation_error 

    translation_errors.append(translation_error)
    rotation_errors.append(rotation_error)

# Plotting errors against noise
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.plot(noise_levels, translation_errors, label="Translation Error")
plt.title("Translation Error vs Noise")
plt.xlabel("Noise Level (meters)")
plt.ylabel("Translation Error (pixels)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(noise_levels, rotation_errors, label="Rotation Error")
plt.title("Rotation Error vs Noise")
plt.xlabel("Noise Level (meters)")
plt.ylabel("Rotation Error (degrees)")
plt.grid(True)

plt.tight_layout()
plt.show()
