import numpy as np
import ezdxf
import os
import cv2
import pickle
import matplotlib.pyplot as plt

# Constants
file_path = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\boot.dxf"
num_points_per_line = 50
noise_std = 0.05
N = 4
output_dir = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\evaluation_data"
scale_factor = 100  # Assuming DXF units are in meters, 0.01 meters = 1cm

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_size = (300, 300)  # Size of the output images
translation_range = (-1, -1)  # Range for translation
rotation_range = (10, 10)  # Range for rotation

def load_lines_from_dxf(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    return [entity for entity in msp if entity.dxftype() == 'LINE']

def generate_noisy_points(lines, num_points=num_points_per_line, noise_std=noise_std):
    all_points = []
    for line in lines:
        start, end = np.array(line.dxf.start)[:2], np.array(line.dxf.end)[:2]
        points_along_edge = np.linspace(start, end, num_points)
        noise = np.random.normal(scale=noise_std, size=(num_points, 2))
        noisy_points = points_along_edge + noise
        all_points.extend(noisy_points)
    return np.array(all_points) * scale_factor  # Apply scaling here

def transform_points(points, dx, dy, phi_degrees):
    phi = np.radians(phi_degrees)
    R = np.array([[np.cos(phi), np.sin(phi)],
                  [-np.sin(phi), np.cos(phi)]])
    rotated_points = np.dot(points, R.T)
    translated_points = rotated_points + np.array([dx, dy])
    return translated_points

def standardize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    max_distance = np.max(np.sqrt(np.sum(centered_points**2, axis=1)))
    normalized_points = centered_points / max_distance
    return normalized_points

def create_image_from_point_cloud(points, image_size):
    image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255  # Initialize to white background
    scaled_points = ((points + 1) / 2 * np.array([image_size[1], image_size[0]])).astype(int)
    for point in scaled_points:
        if 0 <= point[1] < image_size[0] and 0 <= point[0] < image_size[1]:
            image[point[1], point[0], :] = (0, 0, 0)  # Black pixels for points
    return image


def save_data(features, labels, i, output_dir):
    image = create_image_from_point_cloud(features, image_size)
    cv2.imwrite(os.path.join(output_dir, f'image_{i}.png'), image)
    with open(os.path.join(output_dir, f'training_data_{i}.pkl'), 'wb') as f:
        pickle.dump((features, labels), f)

def visualize_and_annotate(output_dir, num_visualizations=4):
    indices = np.random.choice(N, num_visualizations, replace=False)  # Randomly select indices for visualization

    fig, axs = plt.subplots(1, num_visualizations, figsize=(15, 5))

    for j, index in enumerate(indices):
        # Load data
        image_path = os.path.join(output_dir, f'image_{index}.png')
        data_path = os.path.join(output_dir, f'training_data_{index}.pkl')

        with open(data_path, 'rb') as f:
            _, (dx, dy, phi_degrees) = pickle.load(f)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[j].imshow(image)
        axs[j].set_title(f"dx={dx:.2f}, dy={dy:.2f}, rot={phi_degrees:.2f}°")
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    lines = load_lines_from_dxf(file_path)
    original_noisy_points = generate_noisy_points(lines)
    print("Generating training data ...")

    for i in range(N):
        dx, dy = np.random.uniform(*translation_range, 2)
        phi_degrees = np.random.uniform(*rotation_range)

        transformed_points = transform_points(original_noisy_points, dx, dy, phi_degrees)
        standardized_points = standardize_point_cloud(transformed_points)

        save_data(standardized_points, (dx, dy, phi_degrees), i, output_dir)

    print(f"Saved {N} datasets to {output_dir}")
    visualize_and_annotate(output_dir, num_visualizations=4)
