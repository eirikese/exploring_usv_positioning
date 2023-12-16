import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from itertools import cycle

def fit_line_ransac(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    ransac = RANSACRegressor().fit(X, y)
    return ransac.estimator_.coef_[0], ransac.estimator_.intercept_

# Constants
VECTOR_LENGTH = 2.0
AREA_SIZE = 5.0
BUFFER = 0.5 + VECTOR_LENGTH/2  # extra buffer to ensure the square is inside the plot
INNER_AREA = AREA_SIZE - 2 * BUFFER
NUM_POINTS = 100
NOISE_STD = 0.05

# Generate random square corners and edges within the buffered area
center_x = np.random.uniform(BUFFER, BUFFER + INNER_AREA)
center_y = np.random.uniform(BUFFER, BUFFER + INNER_AREA)

half_length = VECTOR_LENGTH / 2

corners = np.array([[center_x - half_length, center_y - half_length],
                   [center_x + half_length, center_y - half_length],
                   [center_x + half_length, center_y + half_length],
                   [center_x - half_length, center_y + half_length]])

# Apply random rotation
rotation_angle = np.random.uniform(0, 2 * np.pi)
rotation_matrix = np.array([
    [np.cos(rotation_angle), -np.sin(rotation_angle)],
    [np.sin(rotation_angle), np.cos(rotation_angle)]
])

rotated_corners = np.dot(corners, rotation_matrix.T)

# Generate noisy points along the edges and clip them
all_points = []
for i in range(4):
    start, end = rotated_corners[i], rotated_corners[(i+1)%4]
    points_along_edge = np.linspace(start, end, NUM_POINTS)
    noise = np.random.normal(scale=NOISE_STD, size=(NUM_POINTS, 2))
    noisy_points = points_along_edge + noise
    noisy_points = np.clip(noisy_points, 0, AREA_SIZE)
    all_points.extend(noisy_points)
all_points = np.array(all_points)

# Fit lines using RANSAC and get intersections
lines = []
for i in range(4):
    subset_points = all_points[i*NUM_POINTS:(i+1)*NUM_POINTS]
    m, b = fit_line_ransac(subset_points)
    lines.append((m, b))

corners_fitted = []
for i in range(4):
    m1, b1 = lines[i]
    m2, b2 = lines[(i+1)%4]
    x_intersection = (b2 - b1) / (m1 - m2)
    y_intersection = m1 * x_intersection + b1
    corners_fitted.append([x_intersection, y_intersection])

corners_fitted = np.array(corners_fitted)

# Visualization
plt.scatter(all_points[:, 0], all_points[:, 1], label="Noisy Points", alpha=0.6)
for i, corner in enumerate(corners_fitted):
    next_corner = corners_fitted[(i+1)%4]
    plt.plot([corner[0], next_corner[0]], [corner[1], next_corner[1]], 'r-', linewidth=2.5)
plt.legend()
plt.title("Recovering Square from Noisy Points using RANSAC")
plt.xlabel("x")
plt.ylabel("y")
plt.axis([0, AREA_SIZE, 0, AREA_SIZE])
plt.grid(True)
plt.show()
