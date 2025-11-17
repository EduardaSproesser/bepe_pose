from re import match
from unittest import case
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class Estimator:
    def __init__(self, cam_params_file, dict_type=cv2.aruco.DICT_4X4_50):
        # --- Load camera parameters ---
        if isinstance(cam_params_file, str):
            loaded = np.load(cam_params_file)
            self.K = np.array(loaded['K'], dtype=np.float64)
            self.D = np.array(loaded['D'], dtype=np.float64)
            self.XI = np.array(loaded['XI'], dtype=np.float64)
        elif isinstance(cam_params_file, dict):
            self.K = np.array(cam_params_file['K'], dtype=np.float64)
            self.D = np.array(cam_params_file['D'], dtype=np.float64)
            self.XI = np.array(cam_params_file['XI'], dtype=np.float64)
        else:
            raise ValueError("cam_params must be a dict or path to npz file")
        print("K:\n", self.K)
        print("D:\n", self.D)    
        print("XI:\n", self.XI)

        # ArUco setup
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # Create DetectorParameters with version compatibility
        try:
            # Newer OpenCV: DetectorParameters is a class
            self.arucoParams = cv2.aruco.DetectorParameters()
        except Exception:
            # Older OpenCV: use factory function
            self.arucoParams = cv2.aruco.DetectorParameters_create()
    
    def undistort_points(self, pts):
        """Undistort points"""
        pts = np.array(pts, dtype=np.float64).reshape(-1, 1, 2)
        undistorted_pts = cv2.undistortPoints(pts, self.K, self.D)
        return undistorted_pts.reshape(-1, 2)
    
    def get_poses(self, csv_file):
        # Load CSV (semicolon-separated)
        data = pd.read_csv(csv_file, sep=';')

        # --- helper functions --------------------
        def parse_corners_field(s):
            if pd.isna(s) or str(s).strip() == '':
                return []
            parts = str(s).split('|')
            markers = []
            for p in parts:
                nums = [x for x in p.split(',') if x.strip() != '']
                try:
                    nums = [float(x) for x in nums]
                except Exception:
                    continue
                if len(nums) % 2 != 0:
                    continue
                pts = np.array(nums, dtype=np.float64).reshape(-1, 2)
                markers.append(pts)
            return markers

        def parse_ids_field(s):
            if pd.isna(s) or str(s).strip() == '':
                return []
            raw = str(s)
            if '|' in raw:
                parts = raw.split('|')
            elif ',' in raw:
                parts = raw.split(',')
            else:
                parts = [raw]
            ids_parsed = []
            for p in parts:
                p = p.strip()
                if p == '':
                    continue
                # Try int conversion first
                try:
                    ids_parsed.append(int(p))
                    continue
                except ValueError:
                    pass

                # If the value is a float-like string e.g. '5.0', convert to float
                # and then to int if it's integral. Otherwise keep as string.
                try:
                    f = float(p)
                except Exception:
                    ids_parsed.append(p)
                    continue

                if abs(f - round(f)) < 1e-8:
                    ids_parsed.append(int(round(f)))
                else:
                    ids_parsed.append(p)

            return ids_parsed
        # ----------------------------------------

        # Extract columns
        n_valid = data['n_valid'].fillna(0).astype(int).tolist()
        corners = data['corners'].apply(parse_corners_field).tolist()
        ids = data['ids'].apply(parse_ids_field).tolist()
        marker_type = data['marker_type'].iloc[0]
        print(f"Marker type: {marker_type}")

        # Undistort corners if needed
        for i in range(len(corners)):
            for j in range(len(corners[i])):
                corners[i][j] = self.undistort_points(corners[i][j])

        # Define marker size (mm)
        marker_size_dict = {"1p": 8.2, "2e": 5.3, "3e": 4.35, "2v": 4.2, "3v": 4.2}
        marker_size = marker_size_dict.get(marker_type)
        print(f"Using marker size: {marker_size} mm")

        # Estimate pose for each image
        poses = []
        for nv, c, i in zip(n_valid, corners, ids):
            if nv == 1:
                # Try to estimate single marker pose if corners/ids available
                # if len(c) > 0 and len(i) > 0:
                #     try:
                #         rvec, tvec = self.estimate_pose_single(c[0], i[0], marker_size)
                #         poses.append({'rvec': rvec, 'tvec': tvec / 0.69, 'ids': [int(i[0]) if not isinstance(i[0], (list, np.ndarray)) else int(i[0][0]) ]})
                #     except Exception:
                #         poses.append({'rvec': None, 'tvec': None, 'ids': []})
                # else:
                    poses.append({'rvec': None, 'tvec': None, 'ids': []})
            elif nv > 1:
                rvecs, tvecs, used_ids = self.estimate_pose_multi(c, i, marker_type)
                # Divide tvec by 0.69, considering it's a list
                tvecs = [tvec / 1 for tvec in tvecs]
                if rvecs is None or len(rvecs) == 0:
                    poses.append({'rvec': None, 'tvec': None, 'ids': []})
                else:
                    poses.append({'rvec': rvecs, 'tvec': tvecs, 'ids': used_ids})
            else:
                poses.append({'rvec': None, 'tvec': None, 'ids': []})

        return poses

    def check_point_variance(self, all_obj_pts):
        cov = np.cov(all_obj_pts.T)
        eigvals = np.linalg.eigvalsh(cov)
        print("Object points covariance eigenvalues:", eigvals)
        print("Condition (max/min):", float(eigvals.max()/max(eigvals.min(), 1e-12)))
        return eigvals


    def save_computed_poses(self, poses, output_csv):
        """Append computed poses to a copy of the CSV file."""
        # Load original CSV
        data = pd.read_csv(output_csv, sep=';')

        # Prepare new columns
        rvecs_list = []
        tvecs_list = []

        for pose in poses:
            # Support dicts produced by get_poses or old tuple format
            if isinstance(pose, dict):
                rvec = pose.get('rvec')
                tvec = pose.get('tvec')
            elif isinstance(pose, (list, tuple)) and len(pose) >= 2:
                rvec, tvec = pose[0], pose[1]
            else:
                rvec, tvec = None, None

            if rvec is None or tvec is None:
                rvecs_list.append('')
                tvecs_list.append('')
            elif (isinstance(rvec, list) or isinstance(rvec, np.ndarray)) and (isinstance(rvec, list) or (isinstance(rvec, np.ndarray) and rvec.ndim == 2)):
                # Multiple markers
                rvecs_str = ','.join([','.join([f"{val:.6f}" for val in rv.flatten()]) for rv in rvec])
                tvecs_str = ','.join([','.join([f"{val:.6f}" for val in tv.flatten()]) for tv in tvec])
                rvecs_list.append(rvecs_str)
                tvecs_list.append(tvecs_str)
            else:
                # Single marker
                rvecs_list.append(','.join([f"{val:.6f}" for val in rvec.flatten()]))
                tvecs_list.append(','.join([f"{val:.6f}" for val in tvec.flatten()]))

        # Add new columns to DataFrame
        data['rvec_est'] = rvecs_list
        data['tvec_est'] = tvecs_list

        # Save to new CSV
        base, ext = os.path.splitext(output_csv)
        new_csv_file = f"{base}_with_poses{ext}"
        data.to_csv(new_csv_file, sep=';', index=False)
        print(f"Saved computed poses to {new_csv_file}")

    def check_order(self, obj_pts, img_pts, id):
        # print(f"\nMarker ID {id}")
        # for i in range(8):
        #     print(f"Corner {i} in 2D  corresponds to 3D point: {obj_pts[i]}")
        pass

    def estimate_pose_single(self, corners, id, marker_size):
        """Estimate pose for a single marker detection."""
        # Define 3D object points for the marker corners in marker frame (MF)
        obj_points = np.array([
            [-marker_size/2,  marker_size/2, 0],
            [ marker_size/2,  marker_size/2, 0],
            [ marker_size/2, -marker_size/2, 0],
            [-marker_size/2, -marker_size/2, 0]
        ], dtype=np.float32)
        # Solve PnP using IPPE_SQUARE method
        retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_points, 
            np.array(corners, dtype=np.float32).reshape(1, 4, 2), 
            np.eye(3, dtype=np.float32),
            np.zeros((1, 5), dtype=np.float32),
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        # Select solution pointing towards camera
        best_idx, max_forward = 0, -np.inf
        for i, rvec_cand in enumerate(rvecs):
            R_cand, _ = cv2.Rodrigues(rvec_cand)
            forward_score = -R_cand[:, 2][2]  # Z pointing towards camera
            if forward_score < max_forward:
                max_forward = forward_score
                best_idx = i

        # rvec/tvec transform SMF -> CF
        R_cf_smf, _ = cv2.Rodrigues(rvecs[best_idx])
        t_cf_smf = tvecs[best_idx].reshape(3, 1)

        Tcf_smf = np.eye(4)
        Tcf_smf[:3, :3] = R_cf_smf
        Tcf_smf[:3, 3] = t_cf_smf.flatten()

        # Get angle around Z axis to determine marker orientation
        angle_z = np.rad2deg(rvecs[best_idx][2][0])
        # Transformation MF -> SMF (known)
        Tmf_smf_dict = { 
            5: np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 3.09], #3.09
                        [0, 0, 0, 1]]), 

            1: np.array([[1, 0, 0, 0],
                        [0, np.cos(np.deg2rad(165)),-np.sin(np.deg2rad(165)), 5.416],
                        [0, np.sin(np.deg2rad(165)), np.cos(np.deg2rad(165)), -4.407],
                        [0, 0, 0, 1]]),

            0: np.array([[1, 0, 0, 0],
                        [0, np.cos(np.deg2rad(15)),-np.sin(np.deg2rad(15)), -5.416],
                        [0, np.sin(np.deg2rad(15)), np.cos(np.deg2rad(15)), 4.407],
                        [0, 0, 0, 1]])
        }
        # Get Tmf_smf for this id
        # Adjust id to int
        id = int(id)

        Tmf_smf = Tmf_smf_dict[id]
        Tcf_mf = Tcf_smf @ np.linalg.inv(Tmf_smf)
        # Extract corrected rvec/tvec
        R_cf_mf = Tcf_mf[:3, :3]
        corrected_tvec = Tcf_mf[:3, 3].reshape(3, 1)
        corrected_rvec, _ = cv2.Rodrigues(R_cf_mf)

        return corrected_rvec, corrected_tvec


    def estimate_pose_multi(self, corners_list, ids_list, marker_type):

        match marker_type:
            case "2e":
                marker_points_dict = {
                    1: np.array([  # first marker
                        [-2.65, 7.976, 3.721],
                        [2.65, 7.976, 3.721],
                        [2.65, 2.857, 5.092],
                        [-2.65, 2.857, 5.092]
                    ], dtype=np.float32),

                    0: np.array([  # second marker
                        [-2.65, -2.857, 5.092],
                        [2.65, -2.857, 5.092],
                        [2.65, -7.976, 3.721],
                        [-2.65, -7.976, 3.721]
                    ], dtype=np.float32)
                }
            
            case "3e":
                marker_points_dict = {
                    2: np.array([
                        [4.623, 2.165, 4.619],
                        [8.806, 2.165, 3.498],
                        [8.806, -2.165, 3.498],
                        [4.623, -2.165, 4.619]
                    ], dtype=np.float32),
                    3: np.array([
                        [-6.264, 6.54, 3.501],
                        [-2.515, 8.705, 3.501],
                        [-0.423, 5.083, 4.622],
                        [-4.173, 2.918, 4.622]
                    ], dtype=np.float32),
                    4: np.array([
                        [-4.187, -2.922, 4.619],
                        [-0.437, -5.087, 4.619],
                        [-2.528, -8.709, 3.498],
                        [-6.278, -6.544, 3.498]
                    ], dtype=np.float32)
                }
            
            case "2v":
                marker_points_dict = {
                    6: np.array([
                        [ -2.973, 5.935, 4.268],
                        [ 0.00, 8.807, 3.498],
                        [ 2.973, 5.935, 4.268],
                        [ 0.00, 3.063, 5.037]
                    ], dtype=np.float32),
                    7: np.array([
                        [-2.973, -5.935, 4.268],
                        [0.00, -3.063, 5.037],
                        [2.973, -5.935, 4.268],
                        [0.00, -8.807, 3.498]
                    ], dtype=np.float32)
                }
            
            case "3v":
                marker_points_dict = {
                    9: np.array([
                        [-4.403, 7.627, 3.498],
                        [-0.392, 6.626, 4.268],
                        [-1.531, 2.652, 5.037],
                        [-5.542, 3.653, 4.268]
                    ], dtype=np.float32),
                    10: np.array([
                        [ -5.539, -3.666, 4.265],
                        [-1.528, -2.665, 5.035],
                        [-0.389, -6.639, 4.265],
                        [-4.40, -7.64, 3.496]
                    ], dtype=np.float32),
                    8: np.array([
                        [3.063, 0.00, 5.037],
                        [5.935, 2.973, 4.268],
                        [8.807, 0.00, 3.498],
                        [5.935, -2.973, 4.268]
                    ], dtype=np.float32)
                }

            case _:
                raise ValueError("Unknown marker type")
        """Estimate pose for multiple marker detections."""

        all_obj_pts = []
        all_img_pts = []
        used_ids = [] 

        # To all ided markers, collect their corners and corresponding 3D points
        for corners, id in zip(corners_list, ids_list):
            id = int(id)
            if id in marker_points_dict:
                obj_pts = marker_points_dict[id]
                img_pts = np.array(corners, dtype=np.float32).reshape(4, 2)

                all_obj_pts.append(obj_pts)
                all_img_pts.append(img_pts)
                used_ids.append(id)

        if len(all_obj_pts) == 0:
            return np.array([]), np.array([]), np.array([])

        all_obj_pts = np.vstack(all_obj_pts).astype(np.float32)
        all_img_pts = np.vstack(all_img_pts).astype(np.float32)
        rvecs, tvecs = [], []
        
        self.check_point_variance(all_obj_pts)

        self.check_order(all_obj_pts, all_img_pts, used_ids)
        _, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                all_obj_pts,
                all_img_pts,
                np.eye(3, dtype=np.float32),
                np.zeros((1, 5), dtype=np.float32),
                flags=cv2.SOLVEPNP_SQPNP
            )
        

        # Get best solution, smaller reproj
        errors = [np.sum(r) for r in reproj]
        best_idx = np.argmin(errors)

        rvecs = rvecs[best_idx]
        tvecs = tvecs[best_idx]

        proj, _ = cv2.projectPoints(all_obj_pts, rvecs, tvecs, np.eye(3, dtype=np.float32), np.zeros((1, 5), dtype=np.float32))

        # plt.figure(figsize=(6,6))
        # plt.scatter(all_obj_pts[:,0], all_obj_pts[:,1], label="3D points")
        # plt.scatter(proj[:,0,0], proj[:,0,1], label="Projected points")
        # plt.legend()
        # plt.title("Projection check")
        # plt.show()

        return rvecs, tvecs, used_ids
    
if __name__ == "__main__":
    # Example usage

    cam_params_file = "camera_params.npz"
    estimator = Estimator(cam_params_file)
    csv_file = "C:\\Users\\eduar\\OneDrive\\Área de Trabalho\\bepe\\codes\\markers\\data\\d50\\results\\corners_2e_2e_1.csv"
    poses = estimator.get_poses(csv_file)
    estimator.save_computed_poses(poses, csv_file)