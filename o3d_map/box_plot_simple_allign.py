import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def generate_asymmetrical_points(edge_length, points_per_edge, missing_edge=False):
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

def apply_transformation(points, translation=np.array([0, 0]), rotation_angle=0, noise_level=0.01):
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array([[c, -s], [s, c]])
    
    transformed_points = np.dot(points, R) + translation
    transformed_points += np.random.normal(0, noise_level, transformed_points.shape)
    
    return transformed_points

def run_icp(source_points, target_points):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.hstack((source_points, np.zeros((source_points.shape[0], 1)))))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.hstack((target_points, np.zeros((target_points.shape[0], 1)))))

    threshold = 0.99
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return reg_p2p.transformation

def main():
    edge_length = 1.0
    points_per_edge = 25
    rotation_angle = np.radians(30)
    xy_translation = [0.5, 0.5]
    missing_edge = False
    noise_level = 0.02
    N = 200

    source_points = generate_asymmetrical_points(edge_length, points_per_edge)
    target_points = generate_asymmetrical_points(edge_length, points_per_edge, missing_edge)

    transformations = []
    for _ in range(N):
        noisy_target_points = apply_transformation(target_points, translation=np.array(xy_translation), rotation_angle=rotation_angle, noise_level=noise_level)
        transformation = run_icp(source_points, noisy_target_points)
        transformations.append(transformation)

    # Extracting the rotation component of the transformations for error analysis
    rotation_errors = np.rad2deg([np.arccos(t[0, 0]) for t in transformations])  # Extracting the cosine inverse of the first element of the rotation matrix

    plt.boxplot(rotation_errors)
    plt.title('THIS IS WRONG (-: Rotation Errors with Noise Level = 0.02')
    plt.ylabel('Error in Degrees')
    plt.show()

main()
