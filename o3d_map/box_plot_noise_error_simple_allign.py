import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def generate_asymmetrical_points(edge_length, points_per_edge, missing_edge=False):
    # Generate points for an asymmetrical shape
    width = edge_length
    height = edge_length * 0.75
    side_x = np.linspace(0, width, points_per_edge, endpoint=False)
    side_y = np.linspace(0, height, points_per_edge, endpoint=False)
    
    points = np.vstack((np.column_stack((side_x, np.zeros_like(side_x))),
                        np.column_stack((np.ones_like(side_y) * width, side_y)),
                        np.column_stack((side_x[::-1], np.ones_like(side_x) * height))))
    if not missing_edge:
        points = np.vstack((points, np.column_stack((np.zeros_like(side_y), side_y[::-1]))))
    
    return points

def apply_transformation(points, translation=np.array([0, 0]), rotation_angle_deg=0, noise_level=0.01):
    # Convert angle to radians
    rotation_angle_rad = np.radians(rotation_angle_deg)
    
    # Define a 2D rotation matrix
    c, s = np.cos(rotation_angle_rad), np.sin(rotation_angle_rad)
    R = np.array([[c, -s], [s, c]])
    
    # Apply rotation and then translation
    transformed_points = np.dot(points, R) + translation
    # Apply noise
    transformed_points += np.random.normal(0, noise_level, transformed_points.shape)
    
    return transformed_points

def calculate_rotation_error_deg(estimated_transformation, expected_rotation_deg):
    # Extract rotation from transformation matrix
    rotation_mat = estimated_transformation[0:2, 0:2]
    estimated_rotation_rad = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
    estimated_rotation_deg = np.degrees(estimated_rotation_rad)

    # Calculate the smallest angle difference
    rotation_error = np.abs((-estimated_rotation_deg - expected_rotation_deg + 180) % 360 - 180)
    return rotation_error

def exponential_fit(x, a, b):
    return a * (np.exp(b * x) - 1)

def main():
    edge_length = 1.0
    points_per_edge = 25
    N = 200  # Number of iterations
    M = 5   # Number of fixed noise levels
    rotation_angle_deg = 10
    xy_translation = [0.5, 0.5]
    missing_edge = False
    noise_min = 0.01
    noise_max = 0.1

    # Fixed noise levels
    noise_levels = np.linspace(noise_min, noise_max, M)
    
    all_rotation_errors = [[] for _ in range(M)]
    all_translation_errors = [[] for _ in range(M)]

    for i, noise_level in enumerate(noise_levels):
        for _ in range(N):
            source_points = generate_asymmetrical_points(edge_length, points_per_edge, missing_edge)
            target_points = generate_asymmetrical_points(edge_length, points_per_edge, missing_edge)
            target_points = apply_transformation(target_points, translation=np.array(xy_translation), rotation_angle_deg=rotation_angle_deg, noise_level=noise_level)

            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(np.hstack((source_points, np.zeros((source_points.shape[0], 1)))))
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(np.hstack((target_points, np.zeros((target_points.shape[0], 1)))))

            threshold = 0.99
            trans_init = np.identity(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())

            rotation_error_deg = calculate_rotation_error_deg(reg_p2p.transformation, rotation_angle_deg)
            translation_error = np.linalg.norm(reg_p2p.transformation[0:2, 3] - np.array(xy_translation))

            all_rotation_errors[i].append(rotation_error_deg)
            all_translation_errors[i].append(translation_error)

    # Plotting box plots
    plt.figure(figsize=(14, 6))

    # Translation Error Box Plot
    plt.subplot(1, 2, 1)
    plt.boxplot(all_translation_errors, labels=[f'{x:.2f}' for x in noise_levels])
    plt.title('Translation Error Box Plot')
    plt.xlabel('Noise Standard Deviation (meters)')
    plt.ylabel('Error (meters)')

    # Rotation Error Box Plot
    plt.subplot(1, 2, 2)
    plt.boxplot(all_rotation_errors, labels=[f'{x:.2f}' for x in noise_levels])
    plt.title('Rotation Error Box Plot')
    plt.xlabel('Noise Standard Deviation (meters)')
    plt.ylabel('Error (degrees)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()