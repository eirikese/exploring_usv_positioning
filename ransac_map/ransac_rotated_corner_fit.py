import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

def fit_line_ransac(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    ransac = RANSACRegressor().fit(X, y)
    return ransac.estimator_.coef_[0], ransac.estimator_.intercept_

# Constants
VECTOR_LENGTH = 1.0
AREA_SIZE = 5.0
NUM_POINTS = 50
NOISE_STD = 0.01

# Generate random corner with a 90-degree orientation
corner_x = np.random.uniform(VECTOR_LENGTH, AREA_SIZE - VECTOR_LENGTH)
corner_y = np.random.uniform(VECTOR_LENGTH, AREA_SIZE - VECTOR_LENGTH)

# Apply a random rotation
rotation_angle = np.random.uniform(0, 2 * np.pi)
rotation_matrix = np.array([
    [np.cos(rotation_angle), -np.sin(rotation_angle)],
    [np.sin(rotation_angle), np.cos(rotation_angle)]
])

# Generate noisy points
points_vector1 = np.array([VECTOR_LENGTH, 0]).reshape(1, 2)
points_vector2 = np.array([0, VECTOR_LENGTH]).reshape(1, 2)

rotated_vector1 = np.dot(points_vector1, rotation_matrix.T) + [corner_x, corner_y]
rotated_vector2 = np.dot(points_vector2, rotation_matrix.T) + [corner_x, corner_y]

points_along_vector1 = np.linspace([corner_x, corner_y], rotated_vector1[0], NUM_POINTS)
points_along_vector2 = np.linspace([corner_x, corner_y], rotated_vector2[0], NUM_POINTS)

noise1 = np.random.normal(scale=NOISE_STD, size=(NUM_POINTS, 2))
noise2 = np.random.normal(scale=NOISE_STD, size=(NUM_POINTS, 2))
noisy_points1 = points_along_vector1 + noise1
noisy_points2 = points_along_vector2 + noise2

# Combine the points
all_points = np.vstack([noisy_points1, noisy_points2])

# Use RANSAC twice to fit two dominant lines
m1, b1 = fit_line_ransac(all_points)
m2 = -1/m1  # Ensure 90-degree orientation
b2 = np.median([point[1] - m2 * point[0] for point in all_points])

# Find intersection of the two RANSAC lines
corner_x_ransac = (b2 - b1) / (m1 - m2)
corner_y_ransac = m1 * corner_x_ransac + b1

# Determine the direction of vectors and extend to desired length
direction1 = np.array([1, m1])
direction1 /= np.linalg.norm(direction1)
vector1_end = np.array([corner_x_ransac, corner_y_ransac]) + VECTOR_LENGTH * direction1

direction2 = np.array([1, m2])
direction2 /= np.linalg.norm(direction2)
vector2_end = np.array([corner_x_ransac, corner_y_ransac]) + VECTOR_LENGTH * direction2

# Visualization
plt.scatter(all_points[:, 0], all_points[:, 1], label="Noisy Points", alpha=0.6)
plt.plot([corner_x_ransac, vector1_end[0]], [corner_y_ransac, vector1_end[1]], 'r-', label="RANSAC Vector 1", linewidth=2.5)
plt.plot([corner_x_ransac, vector2_end[0]], [corner_y_ransac, vector2_end[1]], 'b-', label="RANSAC Vector 2", linewidth=2.5)
plt.legend()
plt.title("Recovering Vectors from Noisy Points using RANSAC")
plt.xlabel("x")
plt.ylabel("y")
plt.axis([0, AREA_SIZE, 0, AREA_SIZE])
plt.grid(True)
plt.show()
