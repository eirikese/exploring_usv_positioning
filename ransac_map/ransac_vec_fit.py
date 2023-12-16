import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# Constants
VECTOR_LENGTH = 10.0
AREA_SIZE = 50.0
NUM_POINTS = 100
NOISE_STD = 0.3

# Generate a random vector within the frame
random_x = np.random.uniform(0, AREA_SIZE)
random_y = np.random.uniform(0, AREA_SIZE)
random_theta = np.random.uniform(0, 2 * np.pi)
random_end_x = random_x + VECTOR_LENGTH * np.cos(random_theta)
random_end_y = random_y + VECTOR_LENGTH * np.sin(random_theta)

# Generate points from this vector with some noise
points_along_vector = np.linspace([random_x, random_y], [random_end_x, random_end_y], NUM_POINTS)
noise = np.random.normal(scale=NOISE_STD, size=(NUM_POINTS, 2))
noisy_points = points_along_vector + noise

# Use RANSAC to robustly fit a line to the noisy points
X = noisy_points[:, 0].reshape(-1, 1)
y = noisy_points[:, 1]
ransac = RANSACRegressor().fit(X, y)

# Find the slope (m) and y-intercept (b) of the RANSAC line
m = ransac.estimator_.coef_[0]

# Calculate the centroid of noisy points
centroid = np.mean(noisy_points, axis=0)

# Calculate the direction vector from the RANSAC line's slope
direction = np.array([1, m])
direction /= np.linalg.norm(direction)

# Determine the vector start and end from the centroid
half_length = VECTOR_LENGTH / 2
vector_start = centroid - half_length * direction
vector_end = centroid + half_length * direction

# Visualization
plt.scatter(noisy_points[:, 0], noisy_points[:, 1], label="Noisy Points", alpha=0.6)
plt.plot([AREA_SIZE / 2, AREA_SIZE / 2 + VECTOR_LENGTH], [AREA_SIZE / 2, AREA_SIZE / 2], 'b--', label="Initial Vector", linewidth=2.5)
plt.plot([vector_start[0], vector_end[0]], [vector_start[1], vector_end[1]], color='red', label="RANSAC Vector", linewidth=2.5)
plt.legend()
plt.title("Recovering Vector from Noisy Points using RANSAC")
plt.xlabel("x")
plt.ylabel("y")
plt.axis([0, AREA_SIZE, 0, AREA_SIZE])
plt.grid(True)
plt.show()
