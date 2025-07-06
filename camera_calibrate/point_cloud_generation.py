import numpy as np
import open3d as o3d
import cv2
import os

current_dir = os.path.dirname(__file__)


def load_camera_intrinsics(stereomap_file):
    cv_file = cv2.FileStorage(stereomap_file, cv2.FILE_STORAGE_READ)

    # Get camera matrix from stereoMap.xml
    camera_matrix = cv_file.getNode("cameraMatrixL").mat()

    # Extract intrinsic parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    cv_file.release()
    return fx, fy, cx, cy


def create_point_cloud(color_file, depth_file, stereomap_file):
    # Load camera intrinsics
    fx, fy, cx, cy = load_camera_intrinsics(stereomap_file)

    # Create Open3D intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480, fx=fx, fy=fy, cx=cx, cy=cy
    )

    # Load color image
    color_raw = o3d.io.read_image(color_file)

    # Load depth data and convert to Open3D format
    depth_raw = o3d.io.read_image(depth_file)  # Load the raw depth data

    # # Convert depth to uint16 format required by Open3D
    depth_scale = 1000.0  # Scale factor to convert to millimeters
    # depth_o3d = o3d.geometry.Image((depth_raw * depth_scale).astype(np.uint16))

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        # depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=9.0,  # Maximum depth in meters
        # convert_rgb_to_intensity=False,
    )

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Flip the orientation to match standard coordinate system
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


def main():
    current_dir = os.path.dirname(__file__)
    stereomap_file = os.path.join(current_dir, "stereoMap.xml")

    # Specify your saved depth and color files
    timestamp = "221394194767100"  # Replace with actual timestamp
    color_file = os.path.join(current_dir, "images", f"color_{timestamp}.jpg")
    depth_file = os.path.join(current_dir, "images", f"depth_vis_{timestamp}.png")

    # Create point cloud
    pcd = create_point_cloud(color_file, depth_file, stereomap_file)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    # Optionally save the point cloud
    output_file = os.path.join(current_dir, "images", f"pointcloud_{timestamp}.ply")
    o3d.io.write_point_cloud(output_file, pcd)


if __name__ == "__main__":
    main()
