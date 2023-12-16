import numpy as np
import open3d as o3d

def load_point_cloud(file_path):
    points = np.load(file_path)
    if points.shape[1] == 2:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, source, target):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def execute_icp(source, target, threshold, trans_init):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p

def draw_registration_result(source, target, transformation):
    source_temp = source.transform(transformation)
    source_temp.paint_uniform_color([1, 0.706, 0])  # Color the source point cloud
    target.paint_uniform_color([0, 0.651, 0.929])  # Color the target point cloud
    
    # Create an axis mesh with a specified size
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=voxel_size * 10, origin=[0,0,0])
    
    # Draw the point clouds along with the axis
    o3d.visualization.draw_geometries([source_temp, target, axis])



def main(source_file, target_file, voxel_size):
    source = load_point_cloud(source_file)
    target = load_point_cloud(target_file)
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, target)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh, voxel_size)
    print("Global registration result:")
    print(result_ransac)

    threshold = voxel_size * 1.5
    result_icp = execute_icp(source_down, target_down, threshold, result_ransac.transformation)
    print("ICP registration result:")
    print(result_icp.transformation)

    draw_registration_result(source_down, target_down, result_icp.transformation)

if __name__ == "__main__":
    voxel_size = 0.1  # Set the voxel size for downsampling
    source_file_path = r'C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\o3d_map\bay_evaluation_data\data_0.npy'
    target_file_path = r'C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\o3d_map\bay_evaluation_data\data_3.npy'
    main(source_file_path, target_file_path, voxel_size)
