'''Description
This Python script demonstrates a method for registering asymmetrical point clouds using the Iterative Closest Point (ICP) algorithm. 
The script generates two sets of 2D points representing asymmetrical shapes and then applies a transformation (translation, rotation, and noise) 
to one of the sets to simulate a real-world scenario. The ICP algorithm is then used to align the transformed point cloud with the original one.

Features
    Asymmetrical Point Generation: Generates points for an asymmetrical shape, with the option to omit one edge, creating two different shapes.
    Point Cloud Transformation: Applies translation, rotation, and Gaussian noise to one of the point clouds to simulate 
    a real-world scenario where the two point clouds are not perfectly aligned.
    ICP Registration: Utilizes the Open3D library's ICP implementation to align the transformed point cloud with the original one. 
    The script calculates the optimal transformation matrix to minimize the distance between corresponding points in the two point clouds.
    Visualization: Before and after the ICP registration, the point clouds are visualized using Matplotlib to show the initial misalignment and the final alignment.
    Transformation Extraction: After registration, the script extracts and prints the translation and rotation components of the transformation matrix, 
    providing insights into how the algorithm aligned the point clouds.'''

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def generate_asymmetrical_points(edge_length, points_per_edge, missing_edge=False):
    # Generate points for an asymmetrical shape
    width = edge_length  # width of the base
    height = edge_length * 0.75  # height is 75% of the edge length
    side_x = np.linspace(0, width, points_per_edge, endpoint=False)
    side_y = np.linspace(0, height, points_per_edge, endpoint=False)
    
    points = np.vstack((np.column_stack((side_x, np.zeros_like(side_x))),
                        np.column_stack((np.ones_like(side_y) * width, side_y)),
                        np.column_stack((side_x[::-1], np.ones_like(side_x) * height))))
    if not missing_edge:
        points = np.vstack((points, np.column_stack((np.zeros_like(side_y), side_y[::-1]))))
    
    return points

def apply_transformation(points, translation=np.array([0, 0]), rotation_angle=0, noise_level=0):
    # Define a 2D rotation matrix
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array([[c, -s], [s, c]])
    
    # Apply rotation and then translation
    transformed_points = np.dot(points, R) + translation
    # Apply noise
    transformed_points += np.random.normal(0, noise_level, transformed_points.shape)
    
    return transformed_points

def get_rotation_translation_from_matrix(matrix):
    # Assuming matrix is a 4x4 homogeneous transformation matrix
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]

    # Calculate the Euler angles from the rotation matrix
    sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  rotation_matrix[1,0] * rotation_matrix[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else:
        x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = 0

    euler_angles = np.array([x, y, z])
    return euler_angles, translation_vector

def main():
    edge_length = 10.0
    points_per_edge = 50  # 100 points for a complete shape with 4 edges
    noise_level = 0.1
    rotation_angle = np.radians(30)
    xy_translation = [100, 100]

    # Generate points for the asymmetrical shape
    source_points = generate_asymmetrical_points(edge_length, points_per_edge)
    target_points = generate_asymmetrical_points(edge_length, points_per_edge, missing_edge=True)
    target_points = apply_transformation(target_points, translation=np.array(xy_translation), rotation_angle=rotation_angle, noise_level=noise_level)

    # Convert to Open3D point cloud objects
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.hstack((source_points, np.zeros((source_points.shape[0], 1)))))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.hstack((target_points, np.zeros((target_points.shape[0], 1)))))

    # Calculate centroids of source and target
    source_centroid = np.mean(np.asarray(source.points), axis=0)
    target_centroid = np.mean(np.asarray(target.points), axis=0)

    # Compute translation needed to align centroids
    translation_to_target_centroid = target_centroid - source_centroid

    # Create a 3D transformation matrix for the translation
    trans_init = np.identity(4)
    trans_init[0:3, 3] = translation_to_target_centroid[0:3]

    # Translate the source point cloud to align centroids
    #source.transform(trans_init)

    # Visualize initial point clouds using plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Centroid-ICP Point Cloud Fitting\n', fontsize=16)

    initial_offset_text = f'Initial Offset: {xy_translation}, Angle: {np.degrees(rotation_angle):.2f}°'
    axs[0].set_title('Before ICP Registration\n' + initial_offset_text)
    axs[0].scatter(np.asarray(source.points)[:, 0], np.asarray(source.points)[:, 1], c='r', label='Source', s=1)
    axs[0].scatter(np.asarray(target.points)[:, 0], np.asarray(target.points)[:, 1], c='g', label='Target', s=1)
    axs[0].legend()
    axs[0].axis('equal')

    # Run ICP registration
    threshold = 0.99  # A threshold for the ICP algorithm
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Apply the transformation to the source points for visualization
    source.transform(reg_p2p.transformation)

    # Extract translation and rotation from the transformation matrix
    transformation = reg_p2p.transformation
    translation_result = transformation[0:3, 3]
    rotation_result = transformation[0:3, 0:3]
    rotation_angle_result = np.arccos((np.trace(rotation_result) - 1) / 2)
    transformation_text = f'Result Offset: {translation_result[:2]}, Angle: {np.degrees(rotation_angle_result):.2f}°'

    # Visualize the point clouds after registration using plt
    axs[1].set_title('After ICP Registration\n' + transformation_text)
    axs[1].scatter(np.asarray(source.points)[:, 0], np.asarray(source.points)[:, 1], c='r', label='Source (Transformed)', s=10)
    axs[1].scatter(np.asarray(target.points)[:, 0], np.asarray(target.points)[:, 1], c='g', label='Target', s=10)
    axs[1].legend()
    axs[1].axis('equal')

    # Show the plots
    plt.show()

    # Print the transformation matrix
    print("Transformation is:")
    print(transformation)

if __name__ == "__main__":
    main()

