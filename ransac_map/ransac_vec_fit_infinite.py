import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression

# Constants
VECTOR_LENGTH = 30.0
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
line_X = np.array([0, AREA_SIZE]).reshape(-1, 1)
line_y_ransac = ransac.predict(line_X)

# Visualization
plt.scatter(noisy_points[:, 0], noisy_points[:, 1], label="Noisy Points", alpha=0.6)
plt.plot([AREA_SIZE / 2, AREA_SIZE / 2 + VECTOR_LENGTH], [AREA_SIZE / 2, AREA_SIZE / 2], 'b--', label="Initial Vector", linewidth=2.5)
plt.plot(line_X, line_y_ransac, color='red', label="RANSAC Vector", linewidth=2.5)
plt.legend()
plt.title("Recovering Vector from Noisy Points using RANSAC")
plt.xlabel("x")
plt.ylabel("y")
plt.axis([0, AREA_SIZE, 0, AREA_SIZE])
plt.grid(True)
plt.show()
