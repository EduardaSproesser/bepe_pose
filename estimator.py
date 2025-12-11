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
            if nv == 1:
                # Try to estimate single marker pose if corners/ids available
                if len(c) > 0 and len(i) > 0:
                    try:
                    # rvec, tvec = self.estimate_pose_single_raw(c[0], marker_size)
                        rvec, tvec = self.estimate_pose_single(c[0], i[0], marker_size)
                        poses.append({'rvec': rvec, 'tvec': tvec / 0.69, 'ids': [int(i[0]) if not isinstance(i[0], (list, np.ndarray)) else int(i[0][0]) ]})
                    except Exception:
                        poses.append({'rvec': None, 'tvec': None, 'ids': []})
                else:
                    poses.append({'rvec': None, 'tvec': None, 'ids': []})
            elif nv > 1:
                rvecs, tvecs, used_ids = self.estimate_pose_multi(c, i, marker_type)
                # Divide tvec by 0.69, considering it's a list
                tvecs = [tvec / 0.69 for tvec in tvecs]
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

        # Select solution pointing towards camera
        best_idx, max_forward = 0, -np.inf
        for i, rvec_cand in enumerate(rvecs):
            R_cand, _ = cv2.Rodrigues(rvec_cand)
            forward_score = -R_cand[:, 2][2]  # Z pointing towards camera
            if forward_score > max_forward:
                max_forward = forward_score
                best_idx = i

        # rvec/tvec
        rvecs = rvecs[best_idx]
        tvecs = tvecs[best_idx].reshape(3, 1)

        return rvecs, tvecs
    
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
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_points, 
            np.array(corners, dtype=np.float32).reshape(1, 4, 2), 
            np.eye(3, dtype=np.float32),
            np.zeros((1, 5), dtype=np.float32),
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        # Select solution pointing towards camera, arrow point towards negative z
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

        # Get angle around Y axis
        angle_x = np.degrees(rvecs[best_idx][0][0])
        # Transformation MF -> SMF (known)
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
        # Get Tmf_smf for this id
        # Adjust id to int
        id = int(id)

        # Debug: print id
        # print(f"Estimating pose for marker ID: {id}")
        Tmf_smf = Tmf_smf_dict[id]
        Tcf_mf = Tcf_smf @ np.linalg.inv(Tmf_smf)
        # Extract corrected rvec/tvec
        R_cf_mf = Tcf_mf[:3, :3]
        corrected_tvec = Tcf_mf[:3, 3].reshape(3, 1)
        corrected_rvec, _ = cv2.Rodrigues(R_cf_mf)

        #Print debug info. Poses before and after correction
        # print(f"Before correction: rvec: {rvecs[best_idx].flatten()}, tvec: {tvecs[best_idx].flatten()}")
        # print(f"After correction: rvec: {corrected_rvec.flatten()}, tvec: {corrected_tvec.flatten()}")

        return corrected_rvec, corrected_tvec


    def estimate_pose_multi_iterative(self, corners_list, ids_list, marker_type):

        # Define marker size (mm)
        marker_size_dict = {"1p": 8.2, "2e": 5.3, "3e": 4.35, "2v": 4.2, "3v": 4.2}
        marker_size = marker_size_dict.get(marker_type)

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

        # To all ided markers, collect their corners and corresponding 3D points
        for corners, id in zip(corners_list, ids_list):
            id = int(id)
            # Get initial pose estimate for each marker using estimate_pose_single_raw
            try:
                # Do this for each id, 8 corners
                rvec_init, tvec_init = self.estimate_pose_single(corners, id, marker_size)
                rvecs_initial.append(rvec_init)
                tvecs_initial.append(tvec_init)
            except Exception:
                pass
            if id in marker_points_dict:
                obj_pts = marker_points_dict[id]
                img_pts = np.array(corners, dtype=np.float32).reshape(4, 2)

                all_obj_pts.append(obj_pts)
                all_img_pts.append(img_pts)
                used_ids.append(id)

        # Create a SINGLE initial estimate using estimate pose single mean
        rvec_init = np.mean(np.array(rvecs_initial), axis=0)
        tvec_init = np.mean(np.array(tvecs_initial), axis=0)

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
            flags=cv2.SOLVEPNP_ITERATIVE # Tenta o Iterative para aproveitar o guess, ou usa para otimização
        )
        
        if not retval:
            print("[ERRO] solvePnP_Iterative falhou.")
            return [], [], used_ids

        # NOVO BLOCO DE PREPARAÇÃO PARA REFINAMENTO ---
        # Garante que os vetores de pose estejam no formato e tipo de dados corretos.
        # O solvePnP pode retornar (3, 1) ou (3,) dependendo da versão/flag. 
        # solvePnPRefineLM prefere (3, 1) e np.float64 para otimização.
        
        rvec_refined = rvec_multi.copy().reshape(3, 1).astype(np.float64)
        tvec_refined = tvec_multi.copy().reshape(3, 1).astype(np.float64)
        # 3. Refinamento Levenberg-Marquardt (LM)
        try:
            rvec_multi, tvec_multi = cv2.solvePnPRefineLM(
                all_obj_pts, 
                all_img_pts, 
                np.eye(3, dtype=np.float32),
                np.zeros((1, 5), dtype=np.float32),
                rvec_refined, 
                tvec_refined, 
            )
        except Exception as e:
            print(f"[AVISO] solvePnPRefineLM falhou. Usando resultado do Iterative. Erro: {e}")

        return rvec_multi, tvec_multi, used_ids
    
    def estimate_pose_multi(self, corners_list, ids_list, marker_type):

        # Define marker size (mm)
        marker_size_dict = {"1p": 8.2, "2e": 5.3, "3e": 4.35, "2v": 4.2, "3v": 4.2}
        marker_size = marker_size_dict.get(marker_type)

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

        # To all ided markers, collect their corners and corresponding 3D points
        for corners, id in zip(corners_list, ids_list):
            id = int(id)
            # Get initial pose estimate for each marker using estimate_pose_single_raw
            try:
                # Do this for each id, 8 corners
                rvec_init, tvec_init = self.estimate_pose_single(corners, id, marker_size)
                rvecs_initial.append(rvec_init)
                tvecs_initial.append(tvec_init)
            except Exception:
                pass
            if id in marker_points_dict:
                obj_pts = marker_points_dict[id]
                img_pts = np.array(corners, dtype=np.float32).reshape(4, 2)

                all_obj_pts.append(obj_pts)
                all_img_pts.append(img_pts)
                used_ids.append(id)

        # Create a SINGLE initial estimate using estimate pose single mean
        rvec_init = np.mean(np.array(rvecs_initial), axis=0)
        tvec_init = np.mean(np.array(tvecs_initial), axis=0)

        # Check we have points
        if not all_obj_pts or not all_img_pts:
            print("[AVISO] Nenhum marcador válido encontrado neste frame para estimativa multi.")
            return [], [], used_ids

        # Stack all points
        all_obj_pts = np.vstack(all_obj_pts).astype(np.float32)
        all_img_pts = np.vstack(all_img_pts).astype(np.float32)
        
    
        return rvec_init, tvec_init, used_ids
    
if __name__ == "__main__":
    # Example usage

    cam_params_file = "camera_params.npz"
    estimator = Estimator(cam_params_file)
    csv_file = "C:\\Users\\eduar\\OneDrive\\Área de Trabalho\\bepe\\codes\\markers\\data\\d50\\results\\corners_2e_2e_2.csv"
    poses = estimator.get_poses(csv_file)
    estimator.save_computed_poses(poses, csv_file)