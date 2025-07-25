# Point Cloud Visualization in Jupyter Notebook
# Run each cell individually for best results

import numpy as np
import open3d as o3d
import cv2
import os
import glob
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Cell 1: Setup and Data Loading
class JupyterPointCloudViewer:
    def __init__(self, data_dir="images"):
        self.data_dir = data_dir
        self.point_cloud_data = None
        self.latest_timestamp = None

    def load_latest_data(self):
        """Load the most recent point cloud data"""
        pkl_files = glob.glob(os.path.join(self.data_dir, "pointcloud_data_*.pkl"))
        if not pkl_files:
            print("No point cloud data files found!")
            return False

        latest_file = max(pkl_files, key=os.path.getctime)
        self.latest_timestamp = latest_file.split("_")[-1].split(".")[0]

        print(f"Loading data from timestamp: {self.latest_timestamp}")

        with open(latest_file, "rb") as f:
            self.point_cloud_data = pickle.load(f)

        print("Available point clouds:")
        for name, data in self.point_cloud_data.items():
            print(f"  {name}: {len(data['points'])} points")

        return True

    def load_camera_intrinsics(self, stereomap_file="stereoMap.xml"):
        """Load camera intrinsics from stereo calibration file"""
        if not os.path.exists(stereomap_file):
            print(f"Warning: {stereomap_file} not found. Using default values.")
            return 500, 500, 320, 240  # Default values

        cv_file = cv2.FileStorage(stereomap_file, cv2.FILE_STORAGE_READ)
        camera_matrix = cv_file.getNode("cameraMatrixL").mat()

        if camera_matrix is not None:
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
        else:
            print("Warning: Could not read camera matrix. Using default values.")
            fx, fy, cx, cy = 500, 500, 320, 240

        cv_file.release()
        return fx, fy, cx, cy

    def create_open3d_pointcloud_from_rgbd(self, pointcloud_name):
        """Create Open3D point cloud using RGBD method (more accurate)"""
        if pointcloud_name not in self.point_cloud_data:
            print(f"Point cloud '{pointcloud_name}' not found!")
            return None

        # Load camera intrinsics
        fx, fy, cx, cy = self.load_camera_intrinsics()

        # Create Open3D intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=640, height=480, fx=fx, fy=fy, cx=cx, cy=cy
        )

        # Load color image
        color_file = os.path.join(
            self.data_dir, f"color_image_{self.latest_timestamp}.jpg"
        )
        if not os.path.exists(color_file):
            print(f"Color image not found: {color_file}")
            return self.create_open3d_pointcloud_simple(pointcloud_name)

        color_raw = o3d.io.read_image(color_file)

        # Get depth data
        data = self.point_cloud_data[pointcloud_name]
        depth_raw = data["depth_map"]

        # Handle depth data
        print(f"Depth statistics for {pointcloud_name}:")
        print(f"  Shape: {depth_raw.shape}")
        valid_mask = (depth_raw > 0) & np.isfinite(depth_raw)
        print(f"  Valid points: {np.sum(valid_mask)}")

        if np.sum(valid_mask) == 0:
            print("No valid depth points found!")
            return None

        print(f"  Min depth: {np.min(depth_raw[valid_mask]):.2f}")
        print(f"  Max depth: {np.max(depth_raw[valid_mask]):.2f}")
        print(f"  Mean depth: {np.mean(depth_raw[valid_mask]):.2f}")

        # Determine appropriate depth scale
        max_depth = np.max(depth_raw[valid_mask])
        if max_depth > 100:  # Likely in mm
            depth_scale = 1000.0  # Convert to meters for Open3D
            depth_trunc = 8.0  # 8 meters max
        else:  # Already in meters
            depth_scale = 1000.0  # Keep in mm for Open3D
            depth_trunc = 8.0

        print(f"Using depth_scale: {depth_scale}, depth_trunc: {depth_trunc}")

        # Convert depth to uint16 format
        depth_scaled = depth_raw * 1000.0 / depth_scale  # Convert to mm
        depth_scaled = np.clip(depth_scaled, 0, 65535)
        depth_o3d = o3d.geometry.Image(depth_scaled.astype(np.uint16))

        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw,
            depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

        # Transform to standard coordinate system
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        print(f"Created point cloud with {len(pcd.points)} points")
        return pcd

    def create_open3d_pointcloud_simple(self, pointcloud_name):
        """Create Open3D point cloud from saved points and colors"""
        if pointcloud_name not in self.point_cloud_data:
            print(f"Point cloud '{pointcloud_name}' not found!")
            return None

        data = self.point_cloud_data[pointcloud_name]
        points = data["points"]
        colors = data["colors"]

        if len(points) == 0:
            print(f"No points found for {pointcloud_name}")
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def preprocess_pointcloud(self, pcd, voxel_size=2.0, remove_outliers=True):
        """Apply preprocessing to point cloud"""
        if pcd is None or len(pcd.points) == 0:
            return pcd

        print(f"Preprocessing point cloud ({len(pcd.points)} points)...")

        if remove_outliers:
            # Remove statistical outliers
            pcd_clean, _ = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            print(f"After outlier removal: {len(pcd_clean.points)} points")
        else:
            pcd_clean = pcd

        # Downsample if too many points
        if len(pcd_clean.points) > 100000:
            pcd_clean = pcd_clean.voxel_down_sample(voxel_size=voxel_size)
            print(f"After downsampling: {len(pcd_clean.points)} points")

        return pcd_clean


# Initialize the viewer
viewer = JupyterPointCloudViewer()
success = viewer.load_latest_data()

if success:
    print("✓ Data loaded successfully!")
else:
    print("✗ Failed to load data. Make sure you have saved point cloud data files.")


# Cell 2: Individual Point Cloud Visualization
def visualize_single_pointcloud(pointcloud_name, use_rgbd=False, preprocess=True):
    """Visualize a single point cloud"""
    if not success:
        print("No data loaded!")
        return

    print(f"\n=== Visualizing {pointcloud_name.upper()} Point Cloud ===")

    # Create point cloud
    if use_rgbd and pointcloud_name in ["sgbm", "final"]:  # Only for depth maps
        pcd = viewer.create_open3d_pointcloud_from_rgbd(pointcloud_name)
    else:
        pcd = viewer.create_open3d_pointcloud_simple(pointcloud_name)

    if pcd is None:
        print("Failed to create point cloud")
        return

    # Preprocess
    if preprocess:
        pcd = viewer.preprocess_pointcloud(pcd)

    # Visualize
    print("Opening visualization window...")
    print("Controls: Mouse drag to rotate, scroll to zoom, right-click drag to pan")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"{pointcloud_name.upper()} Point Cloud",
        width=1200,
        height=800,
    )


# Visualize each point cloud individually
print("\n" + "=" * 50)
print("INDIVIDUAL POINT CLOUD VISUALIZATION")
print("=" * 50)

# Uncomment the ones you want to visualize:
visualize_single_pointcloud("sgbm", use_rgbd=False, preprocess=True)
# visualize_single_pointcloud('midas_aligned', use_rgbd=False, preprocess=True)
# visualize_single_pointcloud('fused', use_rgbd=False, preprocess=True)
# visualize_single_pointcloud('final', use_rgbd=False, preprocess=True)


# Cell 3: Side-by-Side Comparison
def visualize_comparison():
    """Visualize multiple point clouds side by side"""
    if not success:
        print("No data loaded!")
        return

    print("\n=== Point Cloud Comparison ===")

    # Colors for different point clouds
    colors_map = {
        "sgbm": [1, 0, 0],  # Red
        "midas_aligned": [0, 1, 0],  # Green
        "fused": [0, 0, 1],  # Blue
        "final": [1, 1, 0],  # Yellow
    }

    geometries = []
    offset_x = 0
    spacing = 1000  # mm between point clouds

    for name in viewer.point_cloud_data.keys():
        pcd = viewer.create_open3d_pointcloud_simple(name)
        if pcd is None or len(pcd.points) == 0:
            continue

        # Preprocess with more aggressive downsampling for comparison
        pcd_clean = viewer.preprocess_pointcloud(pcd, voxel_size=3.0)

        # Offset for side-by-side display
        points = np.asarray(pcd_clean.points)
        points[:, 0] += offset_x
        pcd_clean.points = o3d.utility.Vector3dVector(points)

        # Set uniform color for comparison
        if name in colors_map:
            colors = np.tile(colors_map[name], (len(points), 1))
            pcd_clean.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(pcd_clean)
        print(f"Added {name}: {len(pcd_clean.points)} points at offset {offset_x}")
        offset_x += spacing

    print("\nColor coding:")
    print("Red: SGBM, Green: MiDaS Aligned, Blue: Fused, Yellow: Final")

    # Visualize
    o3d.visualization.draw_geometries(
        geometries, window_name="Point Cloud Comparison", width=1400, height=900
    )


# Run comparison
visualize_comparison()


# Cell 4: Overlapped Visualization
def visualize_overlapped():
    """Visualize overlapped point clouds in the same space"""
    if not success:
        print("No data loaded!")
        return

    print("\n=== Overlapped Point Clouds ===")

    # Colors for different point clouds (with transparency effect)
    colors_map = {
        "sgbm": [1, 0.2, 0.2],  # Light Red
        "midas_aligned": [0.2, 1, 0.2],  # Light Green
        "fused": [0.2, 0.2, 1],  # Light Blue
        "final": [1, 1, 0.2],  # Light Yellow
    }

    geometries = []

    for name, color in colors_map.items():
        if name not in viewer.point_cloud_data:
            continue

        pcd = viewer.create_open3d_pointcloud_simple(name)
        if pcd is None or len(pcd.points) == 0:
            continue

        # Preprocess with aggressive downsampling for overlap view
        pcd_clean = viewer.preprocess_pointcloud(pcd, voxel_size=5.0)

        # Set uniform color for each point cloud
        points = np.asarray(pcd_clean.points)
        colors = np.tile(color, (len(points), 1))
        pcd_clean.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(pcd_clean)
        print(f"Added {name}: {len(pcd_clean.points)} points")

    print("\nColor coding:")
    print("Light Red: SGBM, Light Green: MiDaS, Light Blue: Fused, Light Yellow: Final")

    # Visualize
    o3d.visualization.draw_geometries(
        geometries, window_name="Overlapped Point Clouds", width=1200, height=800
    )


# Run overlapped visualization
# visualize_overlapped()


# Cell 5: Matplotlib 3D Plots (Alternative visualization)
def matplotlib_visualization(max_points=5000):
    """Create matplotlib 3D plots as alternative visualization"""
    if not success:
        print("No data loaded!")
        return

    print("\n=== Matplotlib 3D Visualization ===")

    colors_map = {
        "sgbm": "red",
        "midas_aligned": "green",
        "fused": "blue",
        "final": "orange",
    }

    # Individual plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={"projection": "3d"})
    axes = axes.flatten()

    plot_idx = 0
    for name, data in viewer.point_cloud_data.items():
        if plot_idx >= 4:
            break

        points = data["points"]
        colors = data["colors"]

        if len(points) == 0:
            continue

        # Subsample for plotting
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points_sample = points[indices]
            colors_sample = colors[indices]
        else:
            points_sample = points
            colors_sample = colors

        ax = axes[plot_idx]

        # Plot with original colors
        scatter = ax.scatter(
            points_sample[:, 0],
            points_sample[:, 1],
            points_sample[:, 2],
            c=colors_sample,
            s=1,
            alpha=0.6,
        )

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"{name.upper()}\n({len(points)} total points)")

        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Comparison plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for name, data in viewer.point_cloud_data.items():
        points = data["points"]

        if len(points) == 0:
            continue

        # Subsample
        if len(points) > 2000:
            indices = np.random.choice(len(points), 2000, replace=False)
            points_sample = points[indices]
        else:
            points_sample = points

        ax.scatter(
            points_sample[:, 0],
            points_sample[:, 1],
            points_sample[:, 2],
            c=colors_map.get(name, "gray"),
            s=1,
            alpha=0.6,
            label=f"{name} ({len(points)} pts)",
        )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Point Cloud Comparison")
    ax.legend()

    plt.show()


# Run matplotlib visualization
# matplotlib_visualization()


# Cell 6: Point Cloud Statistics and Analysis
def analyze_point_clouds():
    """Analyze and compare point cloud statistics"""
    if not success:
        print("No data loaded!")
        return

    print("\n=== Point Cloud Analysis ===")

    stats = {}

    for name, data in viewer.point_cloud_data.items():
        points = data["points"]

        if len(points) == 0:
            continue

        # Basic statistics
        stats[name] = {
            "point_count": len(points),
            "x_range": (points[:, 0].min(), points[:, 0].max()),
            "y_range": (points[:, 1].min(), points[:, 1].max()),
            "z_range": (points[:, 2].min(), points[:, 2].max()),
            "z_mean": points[:, 2].mean(),
            "z_std": points[:, 2].std(),
            "volume_bbox": (
                (points[:, 0].max() - points[:, 0].min())
                * (points[:, 1].max() - points[:, 1].min())
                * (points[:, 2].max() - points[:, 2].min())
            )
            / 1e9,  # Convert to liters
        }

    # Print statistics
    for name, stat in stats.items():
        print(f"\n{name.upper()} Point Cloud:")
        print(f"  Point Count: {stat['point_count']:,}")
        print(f"  X Range: {stat['x_range'][0]:.1f} to {stat['x_range'][1]:.1f} mm")
        print(f"  Y Range: {stat['y_range'][0]:.1f} to {stat['y_range'][1]:.1f} mm")
        print(f"  Z Range: {stat['z_range'][0]:.1f} to {stat['z_range'][1]:.1f} mm")
        print(f"  Mean Depth: {stat['z_mean']:.1f} mm")
        print(f"  Depth Std: {stat['z_std']:.1f} mm")
        print(f"  Bounding Box Volume: {stat['volume_bbox']:.2f} liters")

    # Create comparison plots
    names = list(stats.keys())
    point_counts = [stats[name]["point_count"] for name in names]
    mean_depths = [stats[name]["z_mean"] for name in names]
    volumes = [stats[name]["volume_bbox"] for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Point count comparison
    axes[0].bar(names, point_counts)
    axes[0].set_title("Point Count Comparison")
    axes[0].set_ylabel("Number of Points")
    axes[0].tick_params(axis="x", rotation=45)

    # Mean depth comparison
    axes[1].bar(names, mean_depths)
    axes[1].set_title("Mean Depth Comparison")
    axes[1].set_ylabel("Depth (mm)")
    axes[1].tick_params(axis="x", rotation=45)

    # Volume comparison
    axes[2].bar(names, volumes)
    axes[2].set_title("Bounding Box Volume")
    axes[2].set_ylabel("Volume (liters)")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


# Run analysis
analyze_point_clouds()


# Cell 7: Save Point Clouds in Different Formats
def save_point_clouds():
    """Save point clouds in various formats"""
    if not success:
        print("No data loaded!")
        return

    print("\n=== Saving Point Clouds ===")

    output_dir = os.path.join(viewer.data_dir, "exported_pointclouds")
    os.makedirs(output_dir, exist_ok=True)

    for name in viewer.point_cloud_data.keys():
        pcd = viewer.create_open3d_pointcloud_simple(name)
        if pcd is None or len(pcd.points) == 0:
            continue

        # Preprocess
        pcd_clean = viewer.preprocess_pointcloud(pcd)

        # Save in different formats
        base_name = f"{name}_{viewer.latest_timestamp}"

        # PLY format (most common)
        ply_file = os.path.join(output_dir, f"{base_name}.ply")
        o3d.io.write_point_cloud(ply_file, pcd_clean)

        # PCD format (Point Cloud Data)
        pcd_file = os.path.join(output_dir, f"{base_name}.pcd")
        o3d.io.write_point_cloud(pcd_file, pcd_clean)

        # XYZ format (simple text)
        xyz_file = os.path.join(output_dir, f"{base_name}.xyz")
        o3d.io.write_point_cloud(xyz_file, pcd_clean)

        print(f"Saved {name}: {len(pcd_clean.points)} points")
        print(f"  PLY: {ply_file}")
        print(f"  PCD: {pcd_file}")
        print(f"  XYZ: {xyz_file}")

    print(f"\nAll point clouds saved to: {output_dir}")


# Uncomment to save point clouds
# save_point_clouds()

print("\n" + "=" * 60)
print("JUPYTER NOTEBOOK POINT CLOUD VIEWER")
print("=" * 60)
print("✓ Run each cell individually for different visualizations")
print("✓ Modify the visualization function calls to show/hide different point clouds")
print("✓ Use Open3D controls: drag to rotate, scroll to zoom, right-click to pan")
print("✓ Close visualization windows to continue to next cell")
print("=" * 60)
