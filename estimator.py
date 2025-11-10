from re import match
from unittest import case
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        """Undistort points using the Kannala-Brandt model."""
        pts = np.array(pts, dtype=np.float64).reshape(-1, 1, 2)
        undistorted_pts = cv2.omnidir.undistortPoints(pts, self.K, self.D, self.XI, R=np.eye(3))
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
                rvec, tvec = self.estimate_pose_single(c[0], i[0], marker_size)
                poses.append((rvec, tvec))
            elif nv > 1:
                rvecs, tvecs = self.estimate_pose_multi(c, i, marker_size)
                poses.append((rvecs, tvecs))
            else:
                poses.append((None, None))

        return poses


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

        # rvec/tvec transform SMF -> CF
        R_cf_smf, _ = cv2.Rodrigues(rvecs[0])
        t_cf_smf = tvecs[0].reshape(3, 1)

        Tcf_smf = np.eye(4)
        Tcf_smf[:3, :3] = R_cf_smf
        Tcf_smf[:3, 3] = t_cf_smf.flatten()

        # Transformation MF -> SMF (known)
        Tmf_smf_dict = { 
            5: np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 3.09],
                        [0, 0, 0, 1]]), 

            0: np.array([[1, 0, 0, 0],
                        [0, np.cos(np.deg2rad(15)),-np.sin(np.deg2rad(15)), -5.416],
                        [0, np.sin(np.deg2rad(15)), np.cos(np.deg2rad(15)), 4.407],
                        [0, 0, 0, 1]]),

            1: np.array([[1, 0, 0, 0],
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


    def estimate_pose_multi(self, corners_list, ids_list, marker_size):
        """Estimate pose for multiple marker detections."""
        rvecs, tvecs = [], []
        for corners, id in zip(corners_list, ids_list):
            rvec, tvec = self.estimate_pose_single(corners, id, marker_size)
            rvecs.append(rvec)
            tvecs.append(tvec)
        return rvecs, tvecs
    
if __name__ == "__main__":
    # Example usage

    cam_params_file = "camera_params.npz"
    estimator = Estimator(cam_params_file)
    poses = estimator.get_poses("C:\\Users\\eduar\\OneDrive\\Área de Trabalho\\bepe\\codes\\markers\\data\\d50\\results\\corners_1p_1p_1.csv")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for rvec, tvec in poses:
        #plot poses in 3d grid
        if rvec is not None and tvec is not None:
            ax.quiver(tvec[0], tvec[1], tvec[2], rvec[0], rvec[1], rvec[2], length=0.1)
    plt.show()  