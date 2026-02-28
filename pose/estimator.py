from re import match
from unittest import case
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path

class Estimator:
    def __init__(self, cam_params_file, estimation_type, dict_type=cv2.aruco.DICT_4X4_50):
        self.estimation_type = estimation_type
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
        
        undistorted_pts = cv2.omnidir.undistortPoints(pts, self.K, self.D, xi=self.XI, R=None)
        # undistorted_pts = cv2.omnidir.undistortPoints(pts, self.K, self.D, xi=self.XI, R=np.eye(3))
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
            # print(f"Estimating pose for image with {nv} valid markers.")
            if nv == 1:
                # Try to estimate single marker pose if corners/ids available
                if len(c) > 0 and len(i) > 0:
                    try:
                    # rvec, tvec = self.estimate_pose_single_raw(c[0], marker_size)
                        # print(f"[INFO] Estimating single-marker pose for ID {i[0]}.")
                        rvec, tvec = self.estimate_pose_single(c[0], i[0], marker_size)
                        poses.append({'rvec': rvec, 'tvec': tvec / 0.69, 'ids': [int(i[0]) if not isinstance(i[0], (list, np.ndarray)) else int(i[0][0]) ]})
                    except Exception:
                        poses.append({'rvec': None, 'tvec': None, 'ids': []})
                else:
                    poses.append({'rvec': None, 'tvec': None, 'ids': []})
            elif nv > 1:
                # print(f"[INFO] Estimating multi-marker pose for {nv} markers.")
                try:
                    rvecs, tvecs, used_ids = getattr(self, f"estimate_pose_{self.estimation_type}")(c, i, marker_type)
                    print(f"[INFO] Estimated multi-marker pose with {len(used_ids)} markers.")
                    # Divide tvec by 0.69, considering it's a list
                    tvecs = [tvec / 0.69 for tvec in tvecs]
                    poses.append({'rvec': rvecs, 'tvec': tvecs, 'ids': used_ids})
                except Exception:
                    poses.append({'rvec': None, 'tvec': None, 'ids': []})
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
        new_csv_file = f"{base}_with_poses_{self.estimation_type}{ext}"
        data.to_csv(new_csv_file, sep=';', index=False)
        print(f"Saved computed poses to {new_csv_file}")
    
    def estimate_pose_single_raw(self, corners, marker_size):
        """Estimate pose for a single marker detection."""
        # Define 3D object points for the marker corners in marker frame (MF)
        obj_points = np.array([
            [-marker_size/2,  marker_size/2, 0],
            [ marker_size/2,  marker_size/2, 0],
            [ marker_size/2, -marker_size/2, 0],
            [-marker_size/2, -marker_size/2, 0]
        ], dtype=np.float32)

        # Solve PnP using IPPE_SQUARE method
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_points, 
            np.array(corners, dtype=np.float32).reshape(1, 4, 2), 
            np.eye(3, dtype=np.float32),
            np.zeros((1, 5), dtype=np.float32),
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        return rvecs, tvecs
        
    def _transform_and_correct_pose(self, rvec_smf, tvec_smf, marker_id):
        """
        Transforms the pose estimated in the Marker Frame (MF) to the Sensor/System Frame (SMF)
        and then to the Camera Frame (CF). This resolves the ambiguity from PnP.
        
        Args:
            rvec_smf (np.array): Rotation vector (CF -> SMF) from solvePnP.
            tvec_smf (np.array): Translation vector (CF -> SMF) from solvePnP.
            marker_id (int): ID of the marker to look up the T_mf_smf transformation.

        Returns:
            tuple: (corrected_rvec, corrected_tvec) in the Camera Frame (CF).
        """
        Tmf_smf_dict = { 
            5: np.array([[9.99999940e-01, 1.26880515e-08, 0.00000000e+00, 0.00000000e+00],
                        [1.26880515e-08, 9.99999940e-01, 0.00000000e+00, 0.00000000e+00],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 3.08999991e+00],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 

            1: np.array([[ 1.00000000e+00,  9.76797310e-10,  2.14449414e-09,  0.00000000e+00],
                        [-1.49827084e-09,  9.65955615e-01,  2.58707821e-01,  5.41650009e+00],
                        [-1.81892457e-09, -2.58707821e-01,  9.65955615e-01,  4.40649986e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),

            0: np.array([[ 1.00000000e+00,  3.80532583e-09, -2.34197373e-09,  0.00000000e+00],
                        [-4.28432001e-09,  9.65955615e-01, -2.58707821e-01, -5.41650009e+00],
                        [ 1.27728439e-09,  2.58707821e-01,  9.65955615e-01,  4.40649986e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),

            2: np.array([[ 9.65916097e-01, -6.01903594e-09,  2.58855402e-01,  6.71450043e+00],
                        [ 6.47105480e-09,  1.00000000e+00, -8.94992969e-10,  0.00000000e+00],
                        [-2.58855402e-01,  2.53953658e-09,  9.65916097e-01,  4.05849981e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),

            3: np.array( [[ 8.65997076e-01, -4.83002186e-01, -1.29452437e-01, -3.34375000e+00],
                        [ 5.00049055e-01,  8.36473346e-01,  2.24194840e-01,  5.81149960e+00],
                        [-3.09603865e-06, -2.58884639e-01,  9.65908229e-01,  4.06150007e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),

            4: np.array([[ 8.66036713e-01,  4.82932806e-01, -1.29446656e-01, -3.35749984e+00],
                        [-4.99980509e-01,  8.36508989e-01, -2.24214792e-01, -5.81549978e+00],
                        [ 2.61260084e-06,  2.58899063e-01,  9.65904415e-01,  4.05849981e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
            
            6: np.array([[ 7.07106769e-01, -7.07106769e-01, -9.22171228e-09,  0.00000000e+00],
                        [ 6.83015645e-01,  6.83015704e-01,  2.58803368e-01,  5.93499994e+00],
                        [-1.83001608e-01, -1.83001608e-01,  9.65930045e-01,  4.26775026e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),

            7: np.array([[ 0.70710677, -0.70710677,  0.,  0.        ],
                        [ 0.68301564,  0.68301564, -0.25880337, -5.93500042],
                        [ 0.18300161,  0.18300161,  0.96593004,  4.26774979],
                        [ 0., 0., 0., 1.]]),

            8: np.array([[ 0.68301564, -0.68301564,  0.25880337,  5.93499994],
                        [ 0.70710677,  0.70710677,  0.,          0.        ],
                        [-0.18300161,  0.18300161,  0.96593004,  4.26775026],
                        [ 0.,          0.,          0.,          1.        ]]),

            9: np.array([[ 0.95387042,  0.27090976, -0.12938012, -2.96700001],
                        [-0.23800635,  0.94505203,  0.22411962,  5.13950014],
                        [ 0.18298715, -0.18298778,  0.96593541,  4.26774979],
                        [ 0.,          0.,          0.,          1.        ]]),

            10: np.array([[ 0.95387042, -0.27090973, -0.12938009, -2.96399999],
                        [ 0.23800628,  0.94505209, -0.22411956, -5.15249968],
                        [ 0.18298708,  0.18298773,  0.96593541,  4.26524973],
                        [ 0.,          0.,          0.,          1.]])
        }
        
        # 1. Get the known transformation: T_mf_smf (Marker Frame -> System/Reference Frame)
        Tmf_smf = Tmf_smf_dict.get(marker_id)
        if Tmf_smf is None:
            raise ValueError(f"Tmf_smf not found for marker ID: {marker_id}")

        # 2. Convert rvec/tvec (CF -> SMF) to Homogeneous Matrix (T_cf_smf)
        # R_cf_smf / t_cf_smf is the estimated pose of the SMF in the CF
        R_cf_smf, _ = cv2.Rodrigues(rvec_smf)
        t_cf_smf = tvec_smf.reshape(3, 1)

        Tcf_smf = np.eye(4, dtype=np.float64) # Use float64 for matrix multiplication stability
        Tcf_smf[:3, :3] = R_cf_smf
        Tcf_smf[:3, 3] = t_cf_smf.flatten()

        # 3. Calculate the Final Transformation: T_cf_mf (Camera Frame -> Marker Frame)
        # T_cf_mf = T_cf_smf @ T_smf_mf, where T_smf_mf = inv(T_mf_smf)
        # This gives the pose of the Marker Frame (MF) in the Camera Frame (CF).
        Tcf_mf = Tcf_smf @ np.linalg.inv(Tmf_smf)
        
        # 4. Extract the corrected rvec/tvec
        R_cf_mf = Tcf_mf[:3, :3]
        corrected_tvec = Tcf_mf[:3, 3].reshape(3, 1)
        corrected_rvec, _ = cv2.Rodrigues(R_cf_mf)
        
        return corrected_rvec, corrected_tvec
    
    def estimate_pose_single(self, corners, id, marker_size):
        """Estimate pose for a single marker detection using IPPE_SQUARE, 
        followed by pose correction and selection of the physically valid solution.
        """
        # print(f"Estimating pose single for marker ID {id} with size {marker_size} mm.")
        # 1. Define 3D object points for the marker corners in marker frame (MF)
        obj_points = np.array([
            [-marker_size/2,  marker_size/2, 0],
            [ marker_size/2,  marker_size/2, 0],
            [ marker_size/2, -marker_size/2, 0],
            [-marker_size/2, -marker_size/2, 0]
        ], dtype=np.float32)

        # Ensure ID is integer
        id_val = int(id)
        
        # 2. Solve PnP using IPPE_SQUARE method (returns 2 solutions)
        # Note: IPPE_SQUARE uses Camera Matrix and Dist Coeffs from the class instance
        try:
            _, rvecs_raw, tvecs_raw, _ = cv2.solvePnPGeneric(
                obj_points, 
                np.array(corners, dtype=np.float32).reshape(1, 4, 2), 
                np.eye(3, dtype=np.float32),
                np.zeros((1, 5), dtype=np.float32),
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
        except cv2.error as e:
            # Handle case where PnP fails (e.g., marker partially obscured)
            print(f"[ERROR PnP] solvePnPGeneric failed for ID {id_val}: {e}")
            return None, None
        
        if len(rvecs_raw) < 1:
            return None, None

        # 3. Transform and collect candidate poses
        candidate_poses = []
        for i in range(len(rvecs_raw)):
            rvec_raw = rvecs_raw[i]
            tvec_raw = tvecs_raw[i]

            # Call the new dedicated function to apply transformations
            corrected_rvec, corrected_tvec = self._transform_and_correct_pose(
                rvec_raw, 
                tvec_raw, 
                id_val
            )
            
            # Store the final transformed pose
            candidate_poses.append({
                'rvec': corrected_rvec,
                'tvec': corrected_tvec,
                # Note: Tcf_mf is not needed for the final return, but could be stored here
            })

        # 4. Select the physically valid solution (Ambiguity resolution)
        best_pose = None
        
        # Selection heuristic: Choose the pose where the marker is 'in front' of the camera.
        # This usually means the Z-coordinate (depth) of the marker in the CF is positive.
        for pose in candidate_poses:
            tvec = pose['tvec'].flatten()
            if tvec[2] > 0: # Check if Z-coordinate is positive (in front of camera)
                best_pose = pose
                break
        
        # 5. Final Return
        if best_pose is None:
            # If no solution passes the Z > 0 test (e.g., camera is very close, or flip is not 180), 
            # return the first solution or handle the error.
            print(f"[WARNING] No pose passed Z>0 test for ID {id_val}. Returning first solution.")
            best_pose = candidate_poses[0]
        
        # print(f"[INFO] Selected pose for marker ID {id_val}: rvec={best_pose['rvec'].flatten()}, tvec={best_pose['tvec'].flatten()}")
        return best_pose['rvec'], best_pose['tvec']

    def estimate_pose_multi_iterative(self, corners_list, ids_list, marker_type):
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
        rvecs_initial = []
        tvecs_initial = []
        used_ids = [] 
        # print(f"[INFO] Estimating multi-marker pose for {len(ids_list)} markers of type {marker_type}.")
        # Get initial poses using self.estimate_pose_multi_mean
        rvecs_initial, tvecs_initial, used_ids_mean = self.estimate_pose_multi_mean(corners_list, ids_list, marker_type)
        if rvecs_initial is None or tvecs_initial is None:
            print("[AVISO] estimate_pose_multi_mean não retornou poses iniciais (None). Pulando multi-iterative.")
            return [], [], []
        try:
            print(f"[INFO] Initial poses obtained for {len(rvecs_initial)} markers.")
        except Exception:
            print("[DEBUG] Falha ao imprimir tamanho de rvecs_initial; conteúdo inesperado.")
        if len(rvecs_initial) == 0:
            print("[AVISO] Nenhum marcador válido encontrado neste frame para estimativa multi.")
            return [], [], used_ids
        # To all ided markers, collect their corners and corresponding 3D points
        for corners, id in zip(corners_list, ids_list):
            id = int(id)
            if id in marker_points_dict:
                obj_pts = marker_points_dict[id]
                img_pts = np.array(corners, dtype=np.float32).reshape(4, 2)

                all_obj_pts.append(obj_pts)
                all_img_pts.append(img_pts)
                used_ids.append(id)

        # Create a SINGLE initial estimate using estimate pose single mean
        try:
            rvec_init = np.mean(np.array(rvecs_initial), axis=0)
            tvec_init = np.mean(np.array(tvecs_initial), axis=0)
        except Exception as e:
            print(f"[ERRO] Falha ao computar média das poses iniciais: {e}")
            return [], [], []

        # Check we have points
        if not all_obj_pts or not all_img_pts:
            print("[AVISO] Nenhum marcador válido encontrado neste frame para estimativa multi.")
            return [], [], used_ids

        # Stack all points
        all_obj_pts = np.vstack(all_obj_pts).astype(np.float32)
        all_img_pts = np.vstack(all_img_pts).astype(np.float32)
        
        # self.check_point_variance(all_obj_pts)

        # self.check_order(all_obj_pts, all_img_pts, used_ids)

        # 2. Execução do SOLVEPNP_ITERATIVE
        if rvec_init is not None and tvec_init is not None:
            # Usa o chute do IPPE para o Iterative Solver
            rvec_multi = rvec_init.copy()
            tvec_multi = tvec_init.copy()
            use_guess = True
        else:
            # Se não encontrou o ID 0 (chute), usa uma estimativa inicial padrão (EPNP)
            rvec_multi = None
            tvec_multi = None
            use_guess = False
            print("[AVISO] Chute inicial não encontrado. Usando EPNP como inicializador.")
        
        # O PnP Iterative ou EPNP/Default PnP
        retval, rvec_multi, tvec_multi = cv2.solvePnP(
            all_obj_pts, 
            all_img_pts, 
            np.eye(3, dtype=np.float32),
            np.zeros((1, 5), dtype=np.float32),
            rvec=rvec_multi if use_guess else None, 
            tvec=tvec_multi if use_guess else None,
            useExtrinsicGuess=use_guess, 
            flags=cv2.SOLVEPNP_SQPNP # Tenta o Iterative para aproveitar o guess, ou usa para otimização
        )
        
        if not retval:
            print("[ERRO] solvePnP falhou.")
            return [], [], used_ids
        
        print(f"[INFO] Multi-marker pose estimated successfully.")
        print(f"[INFO] Multi-marker pose estimated with {len(used_ids)} markers.")
        return rvec_multi, tvec_multi, used_ids
    
    def estimate_pose_multi(self, corners_list, ids_list, marker_type):
        """
        Estimates the multi-marker pose by selecting the IPPE solution combination 
        where the marker rays (Tvec -> Rvec axis) do not intersect.
        """

        # Define marker size (mm)
        marker_size_dict = {"1p": 8.2, "2e": 5.3, "3e": 4.35, "2v": 4.2, "3v": 4.2}
        marker_size = marker_size_dict.get(marker_type)
        
        rvecs_initial = [] # Will store lists of [sol1_rvec, sol2_rvec]
        tvecs_initial = [] # Will store lists of [sol1_tvec, sol2_tvec]
        used_ids = [] 

        # To all ided markers, collect their corners and corresponding 3D points
        for corners, id in zip(corners_list, ids_list):
            id = int(id)
            # Get initial pose estimate for each marker using estimate_pose_single
            try:
                rvec_init, tvec_init = self.estimate_pose_single_raw(corners, marker_size)
                # transform and correct both solutions
                corrected_rvec_0, corrected_tvec_0 = self._transform_and_correct_pose(rvec_init[0], tvec_init[0], id)
                corrected_rvec_1, corrected_tvec_1 = self._transform_and_correct_pose(rvec_init[1], tvec_init[1], id)
                rvecs_initial.append([corrected_rvec_0, corrected_rvec_1])
                tvecs_initial.append([corrected_tvec_0, corrected_tvec_1])
                used_ids.append(id)
            except Exception:
                pass
        # Now we have multiple markers, each with 2 possible poses from IPPE
        n_markers = len(rvecs_initial)
        if n_markers == 0:
            print("[AVISO] Nenhum marcador válido encontrado neste frame para estimativa multi.")
            return [], [], used_ids
        # Generate all combinations of solutions (2^n_markers)
        solution_combinations = list(product([0, 1], repeat=n_markers))
        # ------------ OLD SELECTION METHOD BASED ON INTERATIVE -------------
        # Get the best combination base on rvec (most similar to estimate_pose_multi_iterative result)
        best_combination = None
        min_rvec_diff = float('inf')
        # First, get the iterative solution for comparison
        rvec_iter, tvec_iter, _ = self.estimate_pose_multi_iterative(corners_list, ids_list, marker_type)
        if rvec_iter is None:
            print("[AVISO] estimate_pose_multi_iterative falhou. Retornando vazio.")
            return [], [], used_ids
        for combination in solution_combinations:
            rvecs_selected = []
            tvecs_selected = []
            for marker_idx, sol_idx in enumerate(combination):
                rvecs_selected.append(rvecs_initial[marker_idx][sol_idx])
                tvecs_selected.append(tvecs_initial[marker_idx][sol_idx])
            # Average the selected rvecs for comparison
            rvecs_array = np.array(rvecs_selected)
            rvec_mean = np.mean(rvecs_array, axis=0)
            rvec_diff = np.linalg.norm(rvec_mean - rvec_iter)
            # print(f"Combination {combination} has rvec diff: {rvec_diff:.6f}")
            if rvec_diff < min_rvec_diff:
                min_rvec_diff = rvec_diff
                best_combination = combination
        # print(f"Best combination: {best_combination} with rvec diff: {min_rvec_diff:.6f}")
        # With the best combination, tranform the poses to CF
        rvecs_final = []
        tvecs_final = []
        for marker_idx, sol_idx in enumerate(best_combination):
            id = used_ids[marker_idx]
            rvec_raw = rvecs_initial[marker_idx][sol_idx]
            tvec_raw = tvecs_initial[marker_idx][sol_idx]
            rvecs_final.append(rvec_raw)
            tvecs_final.append(tvec_raw)
        
        # Average the final poses
        rvec_final = np.mean(np.array(rvecs_final), axis=0)
        tvec_final = np.mean(np.array(tvecs_final), axis=0)
        # Return the final averaged pose and used IDs

        # ------------ NEW SELECTION METHOD BASED ON MIN CONSISTENCY ERROR -------------
        # Get the best combination base on min consistency error
        # best_error = float('inf')
        # best_combination = None
        # for combination in solution_combinations:
        #     rvecs_selected = []
        #     tvecs_selected = []
        #     for marker_idx, sol_idx in enumerate(combination):
        #         rvecs_selected.append(rvecs_initial[marker_idx][sol_idx])
        #         tvecs_selected.append(tvecs_initial[marker_idx][sol_idx])
        #     # Compute mean pose
        #     rvecs_array = np.array(rvecs_selected)
        #     tvecs_array = np.array(tvecs_selected)
        #     rvec_mean = np.mean(rvecs_array, axis=0)
        #     tvec_mean = np.mean(tvecs_array, axis=0)
        #     # Compute consistency error (sum of squared distances from mean)
        #     rvec_errors = np.linalg.norm(rvecs_array - rvec_mean, axis=1)
        #     tvec_errors = np.linalg.norm(tvecs_array - tvec_mean, axis=1)
        #     total_error = np.sum(rvec_errors**2) + np.sum(tvec_errors**2)
        #     if total_error < best_error:
        #         best_error = total_error
        #         best_combination = combination
        # # Select the best combination
        # rvecs_final = []
        # tvecs_final = []
        # for marker_idx, sol_idx in enumerate(best_combination):
        #     rvecs_final.append(rvecs_initial[marker_idx][sol_idx])
        #     tvecs_final.append(tvecs_initial[marker_idx][sol_idx])
        # # Average the final poses
        # rvec_final = np.mean(np.array(rvecs_final), axis=0)
        # tvec_final = np.mean(np.array(tvecs_final), axis=0)
        
        return rvec_final, tvec_final, used_ids
    
    def estimate_pose_multi_mean(self, corners_list, ids_list, marker_type):
        """Estimate pose for multiple marker detections using mean of single estimates."""

        rvecs = []
        tvecs = []
        used_ids = []
        # print(f"Estimating multi-marker pose for {len(corners_list)} markers.")

        marker_size_dict = {"1p": 8.2, "2e": 5.3, "3e": 4.35, "2v": 4.2, "3v": 4.2}
        marker_size = marker_size_dict.get(marker_type)
        for corners, id in zip(corners_list, ids_list):
            # print(f"Processing marker ID {id} for multi-marker pose estimation.")
            # print(f"corners: {corners}, id: {id}, marker_type: {marker_size}")
            try:
                rvec, tvec = self.estimate_pose_single(corners, id, marker_size)
                # print(f"Marker ID {id}: rvec = {rvec.flatten()}, tvec = {tvec.flatten()}")
                if rvec is not None and tvec is not None:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                    used_ids.append(id)
            except Exception:
                print(f"[ERRO] Falha ao estimar pose para marcador ID {id} na estimativa multi.")
                pass

        if len(rvecs) == 0 or len(tvecs) == 0:
            return None, None, used_ids

        # Compute mean pose
        rvec_mean = np.mean(np.array(rvecs), axis=0)
        tvec_mean = np.mean(np.array(tvecs), axis=0)

        # print(f"[INFO] Multi-marker mean pose: rvec = {rvec_mean.flatten()}, tvec = {tvec_mean.flatten()}")
        return rvec_mean, tvec_mean, used_ids
    
if __name__ == "__main__":
    repo_root = Path(os.environ.get("BEPE_ROOT", Path(__file__).resolve().parents[1]))
    default_results_root = repo_root.parent / "markers" / "data" / "d100" / "results"
    results_root = Path(os.environ.get("BEPE_RESULTS_ROOT", str(default_results_root)))

    cam_params_file = os.environ.get(
        "BEPE_CAMERA_PARAMS_FILE",
        str(Path(__file__).resolve().parent / "camera_params.npz")
    )
    estimation_type = "multi_mean" # Options: "single", "multi_mean", "multi_iterative", "multi"
    marker_type = "2e"  # Options: "1p", "2e", "3e", "2v", "3v"
    estimator = Estimator(cam_params_file, estimation_type=estimation_type)
    
    # Processar automaticamente _1, _2 e _3
    for i in [1, 2, 3]:
        csv_file = results_root / f"corners_{marker_type}_{marker_type}_{i}.csv"
        if not csv_file.exists():
            print(f"[WARN] File not found, skipping: {csv_file}")
            continue
        print(f"\n{'='*60}")
        print(f"Processando {marker_type}_{marker_type}_{i}...")
        print(f"{'='*60}")
        poses = estimator.get_poses(str(csv_file))
        estimator.save_computed_poses(poses, str(csv_file))
        print(f"✓ {marker_type}_{marker_type}_{i} concluído")
    
    print(f"\n{'='*60}")
    print(f"✅ Todos os arquivos foram processados com sucesso!")
    print(f"{'='*60}")