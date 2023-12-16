import numpy as np
import ezdxf
import os
import cv2
import pickle
import matplotlib.pyplot as plt

# Constants
file_path = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\bay_lines.dxf"
output_dir = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\bay_evaluation_data"

num_points_per_line = 50
noise_std = 1
N = 4
scaling_factor = 10  # Scaling factor (meters = 1, m_to_cm = 100)
image_size = (400 * scaling_factor , 400 * scaling_factor)  # Image size in pixels (fixed to 400x400 for scale 1)
translation_range = (-10 * scaling_factor, 10 * scaling_factor) # pixels
rotation_range = (-180, 180) # degrees

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_lines_from_dxf(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    return [entity for entity in msp if entity.dxftype() == 'LINE']

def generate_noisy_points(lines, num_points=num_points_per_line, noise_std=noise_std):
    all_points = []
    for line in lines:
        start, end = np.array(line.dxf.start)[:2], np.array(line.dxf.end)[:2]
        line_length = np.linalg.norm(end - start)
        actual_num_points = max(int(line_length * num_points), 2)  # Ensure at least two points per line
        points_along_edge = np.linspace(start, end, actual_num_points)
        noise = np.random.normal(scale=noise_std, size=(actual_num_points, 2))
        noisy_points = points_along_edge + noise
        all_points.extend(noisy_points)
    return np.array(all_points)

def center_and_scale_points(points, scaling_factor, image_size):
    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Scale the points
    scaled_points = centered_points * scaling_factor
    
    # Translate points to the image center
    image_center = np.array(image_size) // 2
    translated_points = scaled_points + image_center
    
    return translated_points

def create_image_from_point_cloud(points, image_size):
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    for point in points:
        x, y = point.round().astype(int)
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            image[image_size[1] - y - 1, x] = [0, 0, 0]  # Inverting y-axis to match image coordinates
    return image

def save_data(points, image, i, output_dir):
    cv2.imwrite(os.path.join(output_dir, f'image_{i}.png'), image)
    with open(os.path.join(output_dir, f'training_data_{i}.pkl'), 'wb') as f:
        pickle.dump(points, f)

def apply_random_transform(points, translation_range, rotation_range, image_size):
    # Random translation
    translation = np.random.uniform(*translation_range, size=2)
    translated_points = points + translation

    # Random rotation
    angle_degrees = np.random.uniform(*rotation_range)
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    rotated_points = np.dot(
        translated_points - np.array(image_size) // 2, rotation_matrix
    ) + np.array(image_size) // 2

    return rotated_points, translation, angle_degrees

def visualize_images(output_dir, num_images, transformations, removed_info):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'Images with Transformations and Points Removed {removed_info}', fontsize=16)
    for i in range(num_images):
        image_path = os.path.join(output_dir, f'image_{i}.png')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        axs[i // 2, i % 2].imshow(image, cmap='gray')
        axs[i // 2, i % 2].axis('on')  # Turn off axis
        transformation = transformations[i]
        axs[i // 2, i % 2].set_title(
            f"Image {i}\nTranslation: [{transformation[0][0]:.2f}, {transformation[0][1]:.2f}]\nRotation: {transformation[1]:.2f}°"
        )
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust the top to make space for the suptitle
    plt.show()

    
def filter_points(points):
    # Keep only the points where the x coordinate is greater than or equal to 0
    return points[points[:, 0] >= -50]

if __name__ == "__main__":
    lines = load_lines_from_dxf(file_path)
    original_noisy_points = generate_noisy_points(lines)
    filtered_points = filter_points(original_noisy_points)  # Filter points with x < 0
    scaled_and_centered_points = center_and_scale_points(filtered_points, scaling_factor, image_size)
    transformations = []  # To store transformations for plotting

    print("Generating training data ...")
    for i in range(N):
        transformed_points, translation, angle = apply_random_transform(
            scaled_and_centered_points, translation_range, rotation_range, image_size
        )
        filtered_transformed_points = filter_points(transformed_points)  # Filter again after transformation
        image = create_image_from_point_cloud(filtered_transformed_points, image_size)
        save_data(filtered_transformed_points, image, i, output_dir)
        transformations.append((translation, angle))

    print(f"Saved {N} datasets to {output_dir}")
    removed_info = "where x < -50"  # Description of removed points
    visualize_images(output_dir, N, transformations, removed_info)  # Display images with transformations and removed points info