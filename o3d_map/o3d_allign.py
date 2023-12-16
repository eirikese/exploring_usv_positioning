import numpy as np
import open3d as o3d

def draw_registration_result(source, target, transformation):
    source_temp = o3d.geometry.PointCloud()
    source_temp.points = source.points
    target_temp = o3d.geometry.PointCloud()
    target_temp.points = target.points
    source_temp.paint_uniform_color([1, 0, 0])  # Red for source point cloud
    target_temp.paint_uniform_color([0, 1, 0])  # Green for target point cloud
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name="Registration Result")


def main():
    # Create example source and target 2D point clouds.
    # In a real scenario, you would load your data here.
    source_points = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    target_points = np.array([[0.5, 0.1], [1.5, 1.1], [1.5, 0.1], [0.5, 1.1]])

    # Convert to Open3D point cloud objects.
    # Note: We're adding a third column of zeros to make it 3D.
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.hstack((source_points, np.zeros((source_points.shape[0], 1)))))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.hstack((target_points, np.zeros((target_points.shape[0], 1)))))

    # Visualize initial point clouds
    source.paint_uniform_color([1, 0, 0])  # Red
    target.paint_uniform_color([0, 1, 0])  # Green
    o3d.visualization.draw_geometries([source, target], window_name="Initial Alignment")

    # Run ICP registration.
    threshold = 0.9  # Change this threshold according to your point cloud scale
    trans_init = np.identity(4)  # Identity matrix as an initial guess
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Visualize the point clouds after registration
    draw_registration_result(source, target, reg_p2p.transformation)

    # Print the transformation matrix
    print("Transformation is:")
    print(reg_p2p.transformation)

if __name__ == "__main__":
    main()
