from re import match
import time
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
        self.arucoDict = cv2.aruco.getPredefinedDictionary(dict_type)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    def transform_pf_to_cf(self, px,py,pz,pe):
        
        PF0_inCF = np.array([153.865,-137.38,38.24+2.5]) #mm

        # Extra ajust based on marker_type and id
        match self.marker_type:
            case "1p":
                h = 5.08
                ajust = np.array([np.sin(np.pi - np.deg2rad(pe))*h, 0 , np.cos(np.pi - np.deg2rad(pe))*h])
                Rextra = np.eye(3)
            case "2e":
                h = 0.0
                ajust = np.array([3.422, 0.031, 11.038])
                # 15 degree around Y
                a = np.deg2rad(-15)
                Rextra = np.array([[np.cos(a), 0, np.sin(a)],
                                   [0, 1, 0],
                                   [-np.sin(a), 0, np.cos(a)]])
            case "3e":
                h = 0.0
                ajust = np.array([3.169, 3.565, 25.411])
                Rextra = np.eye(3)
            case "2v":
                h = 0.0
                # ajust = np.array([3.277, -0.898, 10.693])
                # # 90 degree around X
                # a = np.deg2rad(-90)
                # Rextra = np.array([[1, 0, 0],
                #                    [0, np.cos(a), -np.sin(a)],
                #                    [0, np.sin(a), np.cos(a)]])
                ajust = np.array([3.277, -0.898, 10.693])
                Rextra = np.eye(3)
            case "3v":
                h = 0.00
                ajust = np.array([0.0, 0.0, 0.0])
                Rextra = np.eye(3)
            case _:
                ajust = np.array([0, 0 , 0])
                Rextra = np.eye(3)

        cfRpf = np.array([[ -1, 0,  0],
                    [ 0, 1,  0],
                    [-0, 0,  -1]])
        
        pe = -1 * np.deg2rad(pe) #marker X axis is opposite of printer y axis which pe is in
        i0Rcmf = np.array([[ 1, 0,  0],
                    [ 0, np.cos(pe),  -np.sin(pe)],
                    [-0, np.sin(pe),  np.cos(pe)]])
        
        i1Ri0 = np.array([[ 1, 0,  0],
                    [ 0, -1,  0],
                    [0, 0,  -1]])
        
        a =np.deg2rad(-90)
        pfRi1 = np.array([[ np.cos(a), -np.sin(a),  0],
                    [ np.sin(a),np.cos(a),0],
                    [0, 0,  1]])

        cfRcmf = cfRpf @ Rextra @ pfRi1 @ i1Ri0 @ i0Rcmf
        rvec_cf, _ = cv2.Rodrigues(cfRcmf)
        rvec_cf = rvec_cf.flatten()  # Convert from (3,1) to (3,) for cleaner CSV


        pz = -1 * pz  #X axis is inverted because relative to Cam F, right handed Print F requires this
        tvec_cf = (cfRpf @ np.array([[py],[px],[pz]])).flatten() + PF0_inCF + ajust

        # print('YO: ',pe)
        # pfRcmf = pfRi1 @ i1Ri0 @ i0Rcmf
        # print( cfRcmf @ np.array([[1],[1],[1]]))

        # print('TVEC: ',tvec_cf)
        return rvec_cf, tvec_cf
    
    def general_pose_estimator(self, img_folder, csv_path, output_csv):
        # --- Load CSV ---
        photos_df = pd.read_csv(csv_path)
        pose_split = photos_df["pose"].str.split(";", expand=True).astype(float)
        photos_df["x"] = pose_split[0]
        photos_df["y"] = pose_split[1]
        photos_df["z"] = pose_split[2]
        photos_df["e"] = pose_split[3]

        marker_size_dict = {"1p":8.2,"2e":5.3,"3e":4.35,"2v":4.2,"3v":4.2}
        self.marker_type = photos_df["marker_type"].iloc[0]
        print(f"[INFO] Marker type: {self.marker_type}")
        self.marker_size = marker_size_dict[self.marker_type]
        print(f"[INFO] Marker size: {self.marker_size} mm")

        # --- Prepare tvec and rvec lists for PF to CF ---
        rvecs_cf = []
        tvecs_cf = []
        pose_pf = []
        for _, row in photos_df.iterrows():
            pose = (row["x"], row["y"], row["z"], row["e"])
            pose_pf.append(pose)
            rvec, tvec = self.transform_pf_to_cf(row["x"], row["y"], row["z"], row["e"])
            rvecs_cf.append(rvec)
            tvecs_cf.append(tvec)

        # --- Find estimated poses from images ---
        all_rvecs, all_tvecs, all_ids = [], [], []
        for idx, row in photos_df.iterrows():
            img_path = os.path.join(img_folder, f"{int(row['img_id']):04d}.jpg")
            image = cv2.imread(img_path)
            
            # if image is None:
                # print(f"[ERROR] Could not load image: {img_path}")
            # else:
            #     print(f"[DEBUG] Loaded image {idx}: {img_path} - Shape: {image.shape}")
            
            rvecs_img, tvecs_img, ids_img = self.process_image(image)
            if rvecs_img is not None and tvecs_img is not None and ids_img is not None:
                # print(f"Imagem {idx}: {n_detected} marcadores detectados, {ids_img} usados para pose")
                all_rvecs.append(rvecs_img)
                all_tvecs.append(tvecs_img)
                all_ids.append(ids_img)
            else:
                # print(f"Imagem {idx}: {n_detected} marcadores detectados, 0 usados para pose")
                all_rvecs.append(np.array([]))
                all_tvecs.append(np.array([]))
                all_ids.append(np.array([]))
        # pf_points (GT) e cf_points (estimado pela sua transformação) já em arrays (N,3)
        # ou pegue primeiro par para inspeção: