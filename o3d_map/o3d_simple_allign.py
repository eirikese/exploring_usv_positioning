import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def draw_registration_result(source, target, transformation):
    source_temp = source.transform(transformation)
    target_temp = target
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      window_name="Registration Result",
                                      width=600,
                                      height=600)

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

def apply_transformation(points, translation=np.array([0, 0]), rotation_angle=0, noise_level=0.01):
    # Define a 2D rotation matrix
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array([[c, -s], [s, c]])
    
    # Apply rotation and then translation
    transformed_points = np.dot(points, R) + translation
    # Apply noise
    transformed_points += np.random.normal(0, noise_level, transformed_points.shape)
    
    return transformed_points

def main():
    edge_length = 1.0
    points_per_edge = 25  # 100 points for a complete shape with 4 edges
    noise_level = 0.01
    noise_min = 0
    noise_max = 0.1
    N = 200
    rotation_angle = np.radians(30)
    xy_translation = [0.5, 0.5] # [0.5, 0.5] for fail: [1.6, 1.6]
    missing_edge = False

    # Generate points for the asymmetrical shape
    source_points = generate_asymmetrical_points(edge_length, points_per_edge)
    target_points = generate_asymmetrical_points(edge_length, points_per_edge, missing_edge)
    target_points = apply_transformation(target_points, translation=np.array(xy_translation), rotation_angle=rotation_angle, noise_level=noise_level)

    # Convert to Open3D point cloud objects and then to numpy arrays for plt
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.hstack((source_points, np.zeros((source_points.shape[0], 1)))))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.hstack((target_points, np.zeros((target_points.shape[0], 1)))))

    # Visualize initial point clouds using plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].scatter(np.asarray(source.points)[:, 0], np.asarray(source.points)[:, 1], c='r', label='Source')
    ax[0].scatter(np.asarray(target.points)[:, 0], np.asarray(target.points)[:, 1], c='g', label='Target')
    ax[0].set_title('Before Registration, noise = ' + str(noise_level) + 'm')
    ax[0].set_xlabel('m')
    ax[0].set_ylabel('m')
    ax[0].legend()
    ax[0].axis('equal')

    # Run ICP registration
    threshold = 0.99  # A threshold for the ICP algorithm
    trans_init = np.identity(4)  # Identity matrix as an initial guess
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    # Apply the transformation to the source points for visualization
    source.transform(reg_p2p.transformation)

    # Visualize the point clouds after registration using plt
    ax[1].scatter(np.asarray(source.points)[:, 0], np.asarray(source.points)[:, 1], c='r', label='Source (Transformed)')
    ax[1].scatter(np.asarray(target.points)[:, 0], np.asarray(target.points)[:, 1], c='g', label='Target')
    ax[1].set_title('After Registration, noise = ' + str(noise_level) + 'm')
    ax[1].set_xlabel('m')
    ax[1].set_ylabel('m')
    ax[1].legend()
    ax[1].axis('equal')

    # Show the plots
    plt.show()

    # Print the transformation matrix
    print("Transformation is:")
    print(reg_p2p.transformation)
    
    # Extract rotation matrix
    R = reg_p2p.transformation[0:2, 0:2]
    # Calculate rotation angle in radians
    angle_rad = np.arctan2(R[1, 0], R[0, 0])
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    # Print rotation angle in degrees
    print("Rotation in degrees:", angle_deg)


if __name__ == "__main__":
    main()