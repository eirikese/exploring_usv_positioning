'''This code is designed for point cloud registration using Open3D, a popular library for 3D data processing. 
Here's a breakdown of its functionality:

Point Generation:
    generate_asymmetrical_points and generate_L_shaped_points create 2D points in specific shapes. 
    The first function creates a rectangle or a U-shape (if missing_edge is True), and the second creates an L-shape. 
    These points serve as the basis for constructing 3D point clouds.
Transformation and Noise Addition:
    apply_transformation applies a rotational and translational transformation to the points, along with some noise.   
    This simulates a real-world scenario where point clouds from different viewpoints need alignment.
Point Cloud Preprocessing:
    preprocess_point_cloud downsamples the point clouds using voxelization and estimates normals, 
    which are crucial for registration algorithms.
Feature Extraction and Global Registration:
    prepare_dataset preprocesses the point clouds and computes Fast Point Feature Histograms (FPFH) for feature-based global registration.
    execute_global_registration performs RANSAC-based global registration using the FPFH features. 
    This step estimates an initial alignment of the point clouds.
Iterative Closest Point (ICP) Registration:
    After obtaining an initial alignment, the script uses the Iterative Closest Point (ICP) algorithm 
    to refine the alignment by minimizing the distance between corresponding points.
Visualization:
    draw_registration_result visualizes the original and transformed point clouds for comparison.
Main Function:
    The main function ties all these steps together.
    It generates L-shaped point clouds, applies transformations to simulate a real-world scenario, 
    and then processes and aligns these point clouds using global registration followed by ICP.'''

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

def generate_L_shaped_points(edge_length, points_per_edge, missing_edge=False):
    # Base edge on X axis
    base_x = np.linspace(0, edge_length, points_per_edge, endpoint=False)
    base_y = np.zeros_like(base_x)
    
    # Vertical edge on Y axis
    vertical_x = np.zeros(points_per_edge // 2)
    vertical_y = np.linspace(0, edge_length / 2, points_per_edge // 2, endpoint=False)
    
    # Combine to make an L-shape
    points = np.vstack((np.column_stack((base_x, base_y)),
                        np.column_stack((vertical_x, vertical_y))))
    return points


def apply_transformation(points, translation=np.array([0, 0]), rotation_angle=0, noise_level=0.01):
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array([[c, -s], [s, c]])
    transformed_points = np.dot(points, R) + translation
    transformed_points += np.random.normal(0, noise_level, transformed_points.shape)
    return transformed_points

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999))
    return result


def prepare_dataset(voxel_size, source, target):
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)
    
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return source_down, target_down, source_fpfh, target_fpfh

def main():
    edge_length = 1.0
    points_per_edge = 25
    noise_level = 0.01
    rotation_angle = np.radians(20)  # larger rotation angle
    xy_translation = [0.5, 0.5]

    # Generate points for the asymmetrical shape
    source_points = generate_L_shaped_points(edge_length, points_per_edge)
    target_points = generate_L_shaped_points(edge_length, points_per_edge, missing_edge=True)
    target_points = apply_transformation(target_points, translation=np.array(xy_translation), rotation_angle=rotation_angle, noise_level=noise_level)

    # Convert to Open3D point cloud objects
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.hstack((source_points, np.zeros((source_points.shape[0], 1)))))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.hstack((target_points, np.zeros((target_points.shape[0], 1)))))

    voxel_size = 0.05  # means 5cm for the dataset
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, target)

    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("Global registration result:")
    print(result_ransac.transformation)

    threshold = 0.5  # 2cm distance threshold for ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    draw_registration_result(source, target, reg_p2p.transformation)
    print("Transformation is:")
    print(reg_p2p.transformation)

if __name__ == "__main__":
    main()