import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FindTransformation:
    def __init__(self, csv_file):
        # Initialize any required variables or configurations
        # Get data from csv file rvecs, tvecs, n_valid, ids, rvec_est and tvec_est
        # The CSV produced by find_corners.py uses semicolons as separators and
        # contains commas inside vector fields (rvec/tvec/corners). Read with
        # sep=';' to avoid tokenization errors.
        self.data = pd.read_csv(csv_file, sep=';')

    def get_transformation(self, id, rvec_smf, tvec_smf, rvec_real, tvec_real):
        # Define static marker to frame transformations for each marker ID
        R_cf_smf, _ = cv2.Rodrigues(rvec_smf)
        t_cf_smf = tvec_smf.reshape(3, 1)

        Tcf_smf = np.eye(4)
        Tcf_smf[:3, :3] = R_cf_smf
        Tcf_smf[:3, 3] = t_cf_smf.flatten()

        # Get the static marker to frame transformation using the real pose
        R_cf_real, _ = cv2.Rodrigues(rvec_real)
        t_cf_real = tvec_real.reshape(3, 1)
        Tcf_real = np.eye(4)
        Tcf_real[:3, :3] = R_cf_real
        Tcf_real[:3, 3] = t_cf_real.flatten()

        # Compute the transformation from static marker frame to real frame
        Tmf_smf = np.linalg.inv(Tcf_smf) @ Tcf_real
        return Tmf_smf
    
    def compute_mean_transformations(self):
        transformations = {}
        counts = {}

        for _, row in self.data.iterrows():
            if pd.isna(row['rvec']) or pd.isna(row['tvec']) or pd.isna(row['rvec_est']) or pd.isna(row['tvec_est']):
                continue
            marker_type = row['marker_type']
            rvec_smf = np.array([float(val) for val in row['rvec'].split(',')])
            tvec_smf = np.array([float(val) for val in row['tvec'].split(',')])
            rvec_real = np.array([float(val) for val in row['rvec_est'].split(',')])
            tvec_real = np.array([float(val) for val in row['tvec_est'].split(',')])

            Tmf = self.get_transformation(marker_type, rvec_smf, tvec_smf, rvec_real, tvec_real)
            if marker_type not in transformations:
                transformations[marker_type] = Tmf
                counts[marker_type] = 1
            else:
                transformations[marker_type] += Tmf
                counts[marker_type] += 1

        # Compute the mean transformations
        mean_transformations = {id: transformations[id] / counts[id] for id in transformations}
        return mean_transformations
    
    def get_tranformation_angles(self, T):
        R = T[:3, :3]
        t = T[:3, 3]

        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)

        angles_x = np.degrees(rvec[0][0])
        angles_y = np.degrees(rvec[1][0])
        angles_z = np.degrees(rvec[2][0])

        return angles_x, angles_y, angles_z, tvec.flatten()

# Example usage:

if __name__ == "__main__":
    csv_file = "C:\\Users\\eduar\\OneDrive\\Área de Trabalho\\bepe\\codes\\markers\\data\\d50\\results\\corners_1p_1p_1_with_poses.csv"
    finder = FindTransformation(csv_file)

    mean_transformations = finder.compute_mean_transformations()
    angles_transformations = {}

    for marker_type, T in mean_transformations.items():
        angles_x, angles_y, angles_z, tvec = finder.get_tranformation_angles(T)
        angles_transformations[id] = (angles_x, angles_y, angles_z, tvec)
        print(f"Marker Type: {marker_type}\nMean Transformation:\n{T}\n")
        print(f"Rotation Angles (degrees): X: {angles_x}, Y: {angles_y}, Z: {angles_z}")
        print(f"Translation Vector: {tvec}\n")


    # Print the mean transformations for each marker ID