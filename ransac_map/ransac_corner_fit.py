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

# Generate a random corner (two perpendicular vectors) within the frame
corner_x = np.random.uniform(VECTOR_LENGTH, AREA_SIZE - VECTOR_LENGTH)
corner_y = np.random.uniform(VECTOR_LENGTH, AREA_SIZE - VECTOR_LENGTH)

# Generate points from these vectors with noise
points_vector1 = np.linspace([corner_x, corner_y], [corner_x + VECTOR_LENGTH, corner_y], NUM_POINTS)
points_vector2 = np.linspace([corner_x, corner_y], [corner_x, corner_y + VECTOR_LENGTH], NUM_POINTS)
noise1 = np.random.normal(scale=NOISE_STD, size=(NUM_POINTS, 2))
noise2 = np.random.normal(scale=NOISE_STD, size=(NUM_POINTS, 2))
noisy_points1 = points_vector1 + noise1
noisy_points2 = points_vector2 + noise2

# Combine the points
all_points = np.vstack([noisy_points1, noisy_points2])

# Fit the first vector using RANSAC
m1, b1 = fit_line_ransac(noisy_points1)

# Fit the second vector using RANSAC
m2, b2 = fit_line_ransac(noisy_points2)

# Find the intersection of the two lines to get the corner point
corner_x_ransac = (b2 - b1) / (m1 - m2)
corner_y_ransac = m1 * corner_x_ransac + b1

# Extend from the intersection point to get the fitted vectors of desired length
vector1_start = np.array([corner_x_ransac, corner_y_ransac])
vector1_end = vector1_start + np.array([VECTOR_LENGTH, 0])

vector2_start = np.array([corner_x_ransac, corner_y_ransac])
vector2_end = vector2_start + np.array([0, VECTOR_LENGTH])

# Visualization
plt.scatter(all_points[:, 0], all_points[:, 1], label="Noisy Points", alpha=0.6)
plt.plot([vector1_start[0], vector1_end[0]], [vector1_start[1], vector1_end[1]], 'r-', label="RANSAC Vector 1", linewidth=2.5)
plt.plot([vector2_start[0], vector2_end[0]], [vector2_start[1], vector2_end[1]], 'b-', label="RANSAC Vector 2", linewidth=2.5)
plt.legend()
plt.title("Recovering Corners from Noisy Points using RANSAC")
plt.xlabel("x")
plt.ylabel("y")
plt.axis([0, AREA_SIZE, 0, AREA_SIZE])
plt.grid(True)
plt.show()
