import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Constants
VECTOR_LENGTH = 10.0
AREA_SIZE = 50.0
NUM_POINTS = 100
NOISE_STD = 0.2

# Function to get the vector's endpoints given its parameters (position and orientation)
def vector_endpoints(params):
    """Calculate the endpoints of the vector given translation and rotation."""
    x, y, theta = params
    x_end = x + VECTOR_LENGTH * np.cos(theta)
    y_end = y + VECTOR_LENGTH * np.sin(theta)
    return np.array([x, y]), np.array([x_end, y_end])

# Objective function to minimize the sum of distances of points to the vector
def objective(params):
    """Objective function to minimize. Calculate the total distance of all points to the vector."""
    start, end = vector_endpoints(params)
    a = end - start
    b = noisy_points - start
    cross_product = np.cross(a, b)
    dot_product = np.dot(a, np.transpose(b))
    distance_to_line = np.abs(cross_product) / np.linalg.norm(a)
    mask = (dot_product >= 0) & (dot_product <= VECTOR_LENGTH**2)
    return np.sum(distance_to_line[mask])

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

# Initial guess (vector initialized in the middle)
initial_guess = [AREA_SIZE / 2, AREA_SIZE / 2, 0]

# Optimize
result = minimize(objective, initial_guess)

# Extract optimized values
optimized_start, optimized_end = vector_endpoints(result.x)

# Visualization
plt.scatter(noisy_points[:, 0], noisy_points[:, 1], label="Noisy Points", alpha=0.6)
plt.plot([AREA_SIZE / 2, AREA_SIZE / 2 + VECTOR_LENGTH], [AREA_SIZE / 2, AREA_SIZE / 2], 'b--', label="Initial Vector", linewidth=2.5)
plt.plot([optimized_start[0], optimized_end[0]], [optimized_start[1], optimized_end[1]], color='red', label="Optimized Vector", linewidth=2.5)
plt.legend()
plt.title("Recovering Vector from Noisy Points")
plt.xlabel("x")
plt.ylabel("y")
plt.axis([0, AREA_SIZE, 0, AREA_SIZE])
plt.grid(True)
plt.show()
