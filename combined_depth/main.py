import numpy as np
import cv2
import torch
import os
from scipy import ndimage
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


class DepthFusion:
    def __init__(self, stereo_config=None):
        self.current_dir = os.path.dirname(__file__)

        # Initialize MiDaS
        self.init_midas()

        # Initialize SGBM with improved parameters
        self.init_sgbm(stereo_config)

        # Load stereo calibration
        self.load_stereo_calibration()

    def init_midas(self):
        """Initialize MiDaS model"""
        self.model_type = "DPT_Large"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform

    def init_sgbm(self, config=None):
        """Initialize SGBM with improved parameters"""
        if config is None:
            # Improved SGBM parameters for denser results
            config = {
                "minDisparity": 0,
                "numDisparities": 64,  # Reduced for speed, increase if needed
                "blockSize": 5,  # Smaller block size for more detail
                "P1": 8 * 1 * 5**2,  # Reduced penalty for small changes
                "P2": 32 * 1 * 5**2,  # Reduced penalty for large changes
                "disp12MaxDiff": 2,
                "uniquenessRatio": 10,  # Reduced for more points
                "speckleWindowSize": 50,  # Reduced for less filtering
                "speckleRange": 1,
                "preFilterCap": 63,
                "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            }

        self.stereo = cv2.StereoSGBM_create(**config)

        # Create WLS filter for post-processing
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.2)

        # Create right matcher for WLS
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)

    def load_stereo_calibration(self):
        """Load stereo calibration parameters"""
        cv_file = cv2.FileStorage(
            os.path.join(self.current_dir, "stereoMap.xml"), cv2.FILE_STORAGE_READ
        )
        self.Left_Stereo_Map_x = cv_file.getNode("stereoMapL_x").mat()
        self.Left_Stereo_Map_y = cv_file.getNode("stereoMapL_y").mat()
        self.Right_Stereo_Map_x = cv_file.getNode("stereoMapR_x").mat()
        self.Right_Stereo_Map_y = cv_file.getNode("stereoMapR_y").mat()
        self.Q = cv_file.getNode("Q").mat()
        cv_file.release()

    def compute_sgbm_depth(self, imgL, imgR):
        """Compute SGBM depth with post-processing"""
        # Convert to grayscale
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Apply stereo rectification
        rectL = cv2.remap(
            grayL, self.Left_Stereo_Map_x, self.Left_Stereo_Map_y, cv2.INTER_LANCZOS4
        )
        rectR = cv2.remap(
            grayR, self.Right_Stereo_Map_x, self.Right_Stereo_Map_y, cv2.INTER_LANCZOS4
        )

        # Apply histogram equalization for better texture
        rectL = cv2.equalizeHist(rectL)
        rectR = cv2.equalizeHist(rectR)

        # Apply Gaussian blur to reduce noise
        rectL = cv2.GaussianBlur(rectL, (3, 3), 0)
        rectR = cv2.GaussianBlur(rectR, (3, 3), 0)

        # Compute disparity maps (left and right for WLS filtering)
        displ = self.stereo.compute(rectL, rectR)
        dispr = self.right_matcher.compute(rectR, rectL)

        # Apply WLS filter
        disparity = self.wls_filter.filter(displ, rectL, None, dispr)
        disparity = disparity.astype(np.float32) / 16.0

        # Convert to depth
        points_3D = cv2.reprojectImageTo3D(disparity, self.Q)
        depth_map = points_3D[:, :, 2]

        # Filter invalid values
        mask = (disparity > 0) & (depth_map > 0) & (depth_map < 5000)
        depth_map[~mask] = 0

        return depth_map, disparity, mask

    def compute_midas_depth(self, img):
        """Compute MiDaS relative depth"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize to 0-1 range
        depth_map = cv2.normalize(
            depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

        return depth_map

    def align_depths_robust(self, sgbm_depth, midas_depth, sgbm_mask):
        """Robustly align MiDaS depth to SGBM scale using RANSAC"""
        valid_points = sgbm_mask & (sgbm_depth > 0)

        if np.sum(valid_points) < 100:  # Need minimum points for alignment
            print("Warning: Not enough valid SGBM points for alignment")
            return midas_depth * 1000  # Return scaled MiDaS

        # Extract valid depth pairs
        sgbm_valid = sgbm_depth[valid_points].reshape(-1, 1)
        midas_valid = midas_depth[valid_points].reshape(-1, 1)

        # Use polynomial features for better fitting
        poly_features = PolynomialFeatures(degree=2)

        # Create RANSAC pipeline
        ransac = Pipeline(
            [
                ("poly", poly_features),
                (
                    "ransac",
                    RANSACRegressor(
                        estimator=None,
                        min_samples=50,
                        residual_threshold=200,  # 200mm tolerance
                        max_trials=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

        try:
            # Fit the model
            ransac.fit(midas_valid, sgbm_valid.ravel())

            # Transform full MiDaS depth
            midas_reshaped = midas_depth.reshape(-1, 1)
            aligned_depth = ransac.predict(midas_reshaped)
            aligned_depth = aligned_depth.reshape(midas_depth.shape)

            # Get inlier mask for quality assessment
            inlier_mask = ransac.named_steps["ransac"].inlier_mask_
            inlier_ratio = np.sum(inlier_mask) / len(inlier_mask)

            print(f"Alignment quality: {inlier_ratio:.2%} inliers")

            return aligned_depth, inlier_ratio

        except Exception as e:
            print(f"Alignment failed: {e}")
            # Fallback to simple linear scaling
            scale = np.median(sgbm_valid / (midas_valid + 1e-6))
            return midas_depth * scale, 0.0

    def fuse_depths(
        self, sgbm_depth, midas_depth_aligned, sgbm_mask, confidence_threshold=0.3
    ):
        """Fuse SGBM and aligned MiDaS depths"""
        h, w = sgbm_depth.shape
        fused_depth = np.zeros_like(sgbm_depth)

        # Compute confidence maps
        sgbm_confidence = self.compute_sgbm_confidence(sgbm_depth, sgbm_mask)
        midas_confidence = (
            np.ones_like(midas_depth_aligned) * 0.8
        )  # MiDaS is generally smooth

        # Create blending weights
        sgbm_weight = sgbm_confidence * sgbm_mask.astype(float)
        midas_weight = midas_confidence * (
            1 - sgbm_mask.astype(float) * 0.5
        )  # Reduce where SGBM exists

        # Normalize weights
        total_weight = sgbm_weight + midas_weight + 1e-6
        sgbm_weight /= total_weight
        midas_weight /= total_weight

        # Fuse depths
        fused_depth = sgbm_weight * sgbm_depth + midas_weight * midas_depth_aligned

        # Use MiDaS where SGBM has no data
        no_sgbm_mask = ~sgbm_mask
        fused_depth[no_sgbm_mask] = midas_depth_aligned[no_sgbm_mask]

        return fused_depth, sgbm_weight, midas_weight

    def compute_sgbm_confidence(self, depth, mask):
        """Compute confidence map for SGBM depth"""
        confidence = np.zeros_like(depth)

        # Local variance as confidence measure
        kernel = np.ones((5, 5), np.float32) / 25
        depth_smooth = cv2.filter2D(depth, -1, kernel)
        variance = cv2.filter2D((depth - depth_smooth) ** 2, -1, kernel)

        # Lower variance = higher confidence
        confidence = np.exp(-variance / 1000)  # Adjust scale as needed
        confidence[~mask] = 0

        return confidence

    def post_process_depth(self, depth_map):
        """Apply post-processing to smooth and fill holes"""
        # Fill small holes
        depth_filled = cv2.medianBlur(depth_map.astype(np.float32), 5)

        # Apply bilateral filter for edge-preserving smoothing
        depth_smooth = cv2.bilateralFilter(
            depth_filled.astype(np.float32), d=9, sigmaColor=50, sigmaSpace=50
        )

        # Morphological closing to fill remaining holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = (depth_smooth > 0).astype(np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Interpolate holes
        if np.sum(mask_closed - mask) > 0:
            # Use inpainting for hole filling
            holes_mask = (mask_closed - mask).astype(np.uint8) * 255
            depth_smooth = cv2.inpaint(
                depth_smooth.astype(np.float32),
                holes_mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA,
            )

        return depth_smooth

    def process_frame(self, imgL, imgR):
        """Process a stereo frame pair"""
        # Compute SGBM depth
        sgbm_depth, disparity, sgbm_mask = self.compute_sgbm_depth(imgL, imgR)

        # Compute MiDaS depth (using left image)
        midas_depth = self.compute_midas_depth(imgL)

        # Align MiDaS to SGBM scale
        midas_aligned, alignment_quality = self.align_depths_robust(
            sgbm_depth, midas_depth, sgbm_mask
        )

        # Fuse the depths
        fused_depth, sgbm_weight, midas_weight = self.fuse_depths(
            sgbm_depth, midas_aligned, sgbm_mask
        )

        # Post-process
        final_depth = self.post_process_depth(fused_depth)

        return {
            "sgbm_depth": sgbm_depth,
            "midas_depth": midas_depth,
            "midas_aligned": midas_aligned,
            "fused_depth": fused_depth,
            "final_depth": final_depth,
            "disparity": disparity,
            "sgbm_mask": sgbm_mask,
            "sgbm_weight": sgbm_weight,
            "midas_weight": midas_weight,
            "alignment_quality": alignment_quality,
        }

    def visualize_results(self, results, min_depth=100, max_depth=3000):
        """Create visualization of all depth maps"""

        def normalize_depth(depth, min_d, max_d):
            depth_norm = np.clip(depth, min_d, max_d)
            return ((depth_norm - min_d) * 255 / (max_d - min_d)).astype(np.uint8)

        # Create visualizations
        viz = {}

        for key in ["sgbm_depth", "midas_aligned", "fused_depth", "final_depth"]:
            if key in results:
                depth_norm = normalize_depth(results[key], min_depth, max_depth)
                viz[key] = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)

        # Visualize weights
        if "sgbm_weight" in results:
            weight_norm = (results["sgbm_weight"] * 255).astype(np.uint8)
            viz["sgbm_weight"] = cv2.applyColorMap(weight_norm, cv2.COLORMAP_JET)

        if "midas_weight" in results:
            weight_norm = (results["midas_weight"] * 255).astype(np.uint8)
            viz["midas_weight"] = cv2.applyColorMap(weight_norm, cv2.COLORMAP_JET)

        return viz


# Example usage
def main():
    # Initialize fusion system
    fusion = DepthFusion()

    # Camera setup (modify IDs as needed)
    CamL_id = 4
    CamR_id = 2

    # Initialize cameras
    if os.name == "nt":
        CamL = cv2.VideoCapture(CamL_id, cv2.CAP_DSHOW)
        CamR = cv2.VideoCapture(CamR_id, cv2.CAP_DSHOW)
    else:
        CamL = cv2.VideoCapture(CamL_id, cv2.CAP_V4L2)
        CamR = cv2.VideoCapture(CamR_id, cv2.CAP_V4L2)

    # Set resolution
    for cam in [CamL, CamR]:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create windows
    cv2.namedWindow("Final Depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)

    while True:
        retL, imgL = CamL.read()
        retR, imgR = CamR.read()

        if retL and retR:
            # Process frame
            results = fusion.process_frame(imgL, imgR)

            # Visualize
            viz = fusion.visualize_results(results)

            # Show final result
            cv2.imshow("Final Depth", viz["final_depth"])

            # Create comparison view
            if "sgbm_depth" in viz and "midas_aligned" in viz:
                comparison = np.hstack(
                    [viz["sgbm_depth"], viz["midas_aligned"], viz["final_depth"]]
                )
                cv2.imshow("Comparison", comparison)

            # Print quality metrics
            print(f"Alignment quality: {results['alignment_quality']:.2%}")
            print(
                f"SGBM coverage: {np.sum(results['sgbm_mask']) / results['sgbm_mask'].size:.2%}"
            )

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("s"):  # Save results
                timestamp = int(time.time())
                for name, data in results.items():
                    if isinstance(data, np.ndarray):
                        if data.dtype == np.bool_:
                            cv2.imwrite(
                                f"images/{name}_{timestamp}.png",
                                data.astype(np.uint8) * 255,
                            )
                        else:
                            np.save(f"images/{name}_{timestamp}.npy", data)
                print(f"Results saved with timestamp {timestamp}")

    # Cleanup
    CamL.release()
    CamR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
