import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import curve_fit

# Constants
VECTOR_LENGTH = 1.0  # meters
AREA_SIZE = 50.0  # meters
NUM_POINTS = 100
NUM_ITERATIONS = 200
MAX_NOISE_STD = 0.1  # meters

def generate_noisy_points(noise_std):
    # Generate a random vector within the frame
    random_x = np.random.uniform(0, AREA_SIZE)
    random_y = np.random.uniform(0, AREA_SIZE)
    random_theta = np.random.uniform(0, 2 * np.pi)
    random_end_x = random_x + VECTOR_LENGTH * np.cos(random_theta)
    random_end_y = random_y + VECTOR_LENGTH * np.sin(random_theta)

    # Generate points from this vector with some noise
    points_along_vector = np.linspace([random_x, random_y], [random_end_x, random_end_y], NUM_POINTS)
    noise = np.random.normal(scale=noise_std, size=(NUM_POINTS, 2))
    noisy_points = points_along_vector + noise

    return noisy_points, random_theta

def calculate_translation_error(noisy_points, ransac):
    X = noisy_points[:, 0].reshape(-1, 1)
    y = noisy_points[:, 1]
    predicted_y = ransac.predict(X)
    error = np.median(np.abs(predicted_y - y))
    return error

def angle_difference(angle1, angle2):
    # Calculate the smallest angle difference (in radians)
    diff = np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))
    if abs(diff) > np.pi/2:
        diff = abs(diff) - np.pi
    return abs(diff)

def calculate_rotation_error(ransac, actual_theta):
    m = ransac.estimator_.coef_[0]
    predicted_theta = np.arctan(m)
    error = angle_difference(predicted_theta, actual_theta)
    return np.degrees(error)  # Convert radians to degrees

# Arrays to store errors
translation_errors = []
rotation_errors = []

# Experiment with varying noise levels
noise_levels = np.linspace(0, MAX_NOISE_STD, NUM_ITERATIONS)

for noise_std in noise_levels:
    noisy_points, actual_theta = generate_noisy_points(noise_std)
    ransac = RANSACRegressor().fit(noisy_points[:, 0].reshape(-1, 1), noisy_points[:, 1])

    trans_error = calculate_translation_error(noisy_points, ransac)
    rotation_error = calculate_rotation_error(ransac, actual_theta)

    translation_errors.append(trans_error)
    rotation_errors.append(rotation_error)

# Fitting linear and exponential trends using median
def linear_trend(x, a, b):
    return a * x + b

def exp_trend(x, a, b):
    return a * np.exp(b * x)

# Curve fitting
lin_params, _ = curve_fit(linear_trend, noise_levels, translation_errors)
exp_params, _ = curve_fit(exp_trend, noise_levels, rotation_errors)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Translation Error Plot
axs[0].plot(noise_levels, translation_errors, label="Translation Error")
axs[0].plot(noise_levels, linear_trend(noise_levels, *lin_params), '--', color='orange', label="Median, linear trend")

# Create a colormap instance
#colormap = plt.get_cmap('RdYlGn_r')

# Translation Error Plot
#colors = colormap(translation_errors / max(translation_errors))  # Normalize and map the noise levels to colors
#for i in range(len(noise_levels) - 1):
#    axs[0].scatter(noise_levels[i:i+2], translation_errors[i:i+2], color=colors[i], marker='o')


axs[0].set_title("Translation Error with Varying Noise Levels")
axs[0].set_xlabel("Noise Standard Deviation (meters)")
axs[0].set_ylabel("Median Absolute Error (meters)")
axs[0].legend()

# Rotation Error Plot
axs[1].plot(noise_levels, rotation_errors, label="Rotation Error")
axs[1].plot(noise_levels, exp_trend(noise_levels, *exp_params), '--', color='orange', label="Median, exponential trend")
axs[1].set_title("Rotation Error with Varying Noise Levels")
axs[1].set_xlabel("Noise Standard Deviation (meters)")
axs[1].set_ylabel("Rotation Error (degrees)")
axs[1].grid(True)

plt.tight_layout()
plt.show()

