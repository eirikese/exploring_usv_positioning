import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Constants
VECTOR_LENGTH = 1.0
AREA_SIZE = 5.0
NUM_POINTS = 100

# Generate random points
points = np.random.rand(NUM_POINTS, 2) * AREA_SIZE

def vector_endpoints(params):
    """Calculate the endpoints of the vector given translation and rotation."""
    x, y, theta = params
    x_end = x + VECTOR_LENGTH * np.cos(theta)
    y_end = y + VECTOR_LENGTH * np.sin(theta)
    return np.array([x, y]), np.array([x_end, y_end])

def objective(params):
    """Objective function to minimize. Calculate the total distance of all points to the vector."""
    start, end = vector_endpoints(params)
    a = end - start
    b = points - start
    cross_product = np.cross(a, b)
    dot_product = np.dot(a, np.transpose(b))
    distance_to_line = np.abs(cross_product) / np.linalg.norm(a)
    mask = (dot_product >= 0) & (dot_product <= VECTOR_LENGTH**2)
    return np.sum(distance_to_line[mask])

# Initial guess
initial_guess = [AREA_SIZE / 2, AREA_SIZE / 2, 0]

# Optimize
result = minimize(objective, initial_guess)

# Extract optimized values
optimized_start, optimized_end = vector_endpoints(result.x)

# Visualization
plt.scatter(points[:, 0], points[:, 1], label="Random Points", alpha=0.6)
plt.plot([optimized_start[0], optimized_end[0]], [optimized_start[1], optimized_end[1]], color='red', label="Optimized Vector", linewidth=2.5)
plt.legend()
plt.title("Best Fit Vector to Random Points")
plt.xlabel("x")
plt.ylabel("y")
plt.axis([0, AREA_SIZE, 0, AREA_SIZE])
plt.grid(True)
plt.show()
