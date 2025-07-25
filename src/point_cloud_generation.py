import numpy as np
import cv2
import os
import open3d as o3d

# Load depth_raw.npy
current_dir = os.path.dirname(__file__)
depth_file = os.path.join(current_dir, "images", "depth_raw.npy")
depth_map = np.load(depth_file)
print("Loaded depth_map shape:", depth_map.shape)

# Load Q matrix from stereoMap.xml
cv_file = cv2.FileStorage(
    os.path.join(current_dir, "stereoMap.xml"), cv2.FILE_STORAGE_READ
)
Q = cv_file.getNode("Q").mat()
cv_file.release()
print("Loaded Q matrix shape:", Q.shape)

# Create 2D grid of pixel coordinates
h, w = depth_map.shape
x, y = np.meshgrid(np.arange(w), np.arange(h))
points_3d = cv2.reprojectImageTo3D(depth_map, Q)
print("Reprojected points_3d shape:", points_3d.shape)

# Filter out invalid points (where depth is 0)
mask = depth_map > 0
points_3d = points_3d[mask]
colors = np.zeros_like(
    points_3d
)  # Placeholder for colors (can be enhanced with RGB later)

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors if added

# Visualize the point cloud
print("Visualizing point cloud...")
o3d.visualization.draw_geometries([pcd], width=1280, height=720)
print("Visualization closed.")
