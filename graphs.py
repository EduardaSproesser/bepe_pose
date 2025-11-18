import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DrawGraphs:
    def __init__(self, csv_file):
        # Initialize any required variables or configurations
        # Get data from csv file rvecs, tvecs, n_valid, ids, rvec_est and tvec_est
        # The CSV produced by find_corners.py uses semicolons as separators and
        # contains commas inside vector fields (rvec/tvec/corners). Read with
        # sep=';' to avoid tokenization errors.
        self.data = pd.read_csv(csv_file, sep=';')

        # Fix bins values for distance and angle
        self.angle_bins = np.arange(0, np.pi/2 + np.deg2rad(5), np.deg2rad(5))
        self.distance_bins = np.arange(0, 300 + 25, 25)

    def plot_3d_poses(self):
        # Plot real and estimates poses in 3D grid (only points)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for index, row in self.data.iterrows():
            # Real poses
            rvec_str = row['rvec']
            tvec_str = row['tvec']
            if pd.notna(rvec_str) and pd.notna(tvec_str):
                rvec = np.array([float(val) for val in rvec_str.split(',')])
                tvec = np.array([float(val) for val in tvec_str.split(',')])
                ax.scatter(tvec[0], tvec[1], tvec[2], c='b', marker='o', label='Real' if index == 0 else "")
            
            # Estimated poses
            rvec_est_str = row['rvec_est']
            tvec_est_str = row['tvec_est']
            if pd.notna(rvec_est_str) and pd.notna(tvec_est_str):
                rvec_est = np.array([float(val) for val in rvec_est_str.split(',')])
                tvec_est = np.array([float(val) for val in tvec_est_str.split(',')])
                ax.scatter(tvec_est[0], tvec_est[1], tvec_est[2], c='r', marker='^', label='Estimated' if index == 0 else "")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
    
    def plot_distance_difference_with_trend(self, abs_diff=False, show_r2=True):
        """
        Plota a diferença de distância (dist - dist_est) vs a distância real dist,
        com linha de tendência linear.

        Parâmetros:
        - abs_diff: se True plota |dist - dist_est| no eixo Y (diferença absoluta).
        - show_r2: se True adiciona o R^2 no título.
        Retorna: (slope, intercept, r2)
        """
        # helper: parse a vector string "x,y,z" or accept already-splitted lists/arrays
        def parse_vec(s):
            if pd.isna(s):
                return None
            if isinstance(s, (list, tuple, np.ndarray)):
                arr = np.array(s, dtype=float)
                if arr.size == 3:
                    return arr
                return None
            try:
                parts = [p for p in str(s).split(',') if p.strip() != '']
                if len(parts) != 3:
                    return None
                return np.array([float(p) for p in parts], dtype=float)
            except Exception:
                return None

        dists = []
        deltas = []

        for idx, row in self.data.iterrows():
            # somente entradas com detecção válida
            try:
                if 'n_valid' in row and (pd.isna(row['n_valid']) or float(row['n_valid']) <= 0):
                    continue
            except Exception:
                # se n_valid não for numérico, tenta converter; caso falhe, assume válido
                pass

            tvec = parse_vec(row.get('tvec', None))
            tvec_est = parse_vec(row.get('tvec_est', None))
            if tvec is None or tvec_est is None:
                continue

            delta = np.linalg.norm(tvec - tvec_est)
            dist = np.linalg.norm(tvec)

            dists.append(dist)
            deltas.append(delta)

        if len(dists) == 0:
            print("[WARN] Nenhum dado válido encontrado para plotar.")
            return None, None, None

        x = np.array(dists)
        y = np.array(deltas)

        # Fit linear (y = m*x + b)
        m, b = np.polyfit(x, y, 1)
        y_fit = m * x + b

        # R^2
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=30, alpha=0.7, label='Pontos (dist, Δdist)')
        # plot trendline sorted by x for a clean line
        order = np.argsort(x)
        ax.plot(x[order], y_fit[order], color='red', linewidth=2, label=f'Trend (y = {m:.3e} x + {b:.3e})')

        ax.set_xlabel('Distance to origin (units)')
        ax.set_ylabel('Absolute difference |dist - dist_est| (units)' if abs_diff else 'Signed difference (dist - dist_est) (units)')
        title = 'Difference of distance vs Distance'
        if show_r2:
            title += f' — R² = {r2:.4f}'
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.show()

        return m, b, r2

    def plot_translation_errors_distance_bins(self):
        distances = []
        errors = []

        for index, row in self.data.iterrows():
            if pd.notna(row['tvec']) and pd.notna(row['tvec_est']):
                tvec = np.array([float(val) for val in row['tvec'].split(',')])
                tvec_est = np.array([float(val) for val in row['tvec_est'].split(',')])
                distance = np.linalg.norm(tvec)
                error = np.linalg.norm(tvec - tvec_est)
                distances.append(distance)
                errors.append(error)

        # Agrupar por bins de distância
        bins = self.distance_bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_errors = []

        for i in range(len(bins) - 1):
            in_bin = [(d, e) for d, e in zip(distances, errors) if bins[i] <= d < bins[i + 1]]
            if len(in_bin) > 0:
                mean_errors.append(np.mean([e for _, e in in_bin]))
            else:
                mean_errors.append(np.nan)

        plt.bar(bin_centers, mean_errors, width=(self.distance_bins[1] - self.distance_bins[0]) * 0.9, color='steelblue', edgecolor='black')
        plt.xlabel('Distance to Origin (units)')
        plt.ylabel('Mean Translation Error (units)')
        plt.title('Translation Error vs Distance')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def plot_rotation_errors_distance_bins(self):
        distances = []
        errors = []

        for index, row in self.data.iterrows():
            if pd.notna(row['rvec']) and pd.notna(row['rvec_est']) and pd.notna(row['tvec']):
                rvec = np.array([float(val) for val in row['rvec'].split(',')])
                rvec_est = np.array([float(val) for val in row['rvec_est'].split(',')])
                tvec = np.array([float(val) for val in row['tvec'].split(',')])
                distance = np.linalg.norm(tvec)
                # Rotation error as angle between vectors
                rvec_n = rvec / np.linalg.norm(rvec)
                rvec_est_n = rvec_est / np.linalg.norm(rvec_est)
                error = np.arccos(np.clip(np.dot(rvec_n, rvec_est_n), -1.0, 1.0))
                distances.append(distance)
                errors.append(error)

        bins = self.distance_bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_errors = []

        for i in range(len(bins) - 1):
            in_bin = [(d, e) for d, e in zip(distances, errors) if bins[i] <= d < bins[i + 1]]
            if len(in_bin) > 0:
                mean_errors.append(np.mean([e for _, e in in_bin]))
            else:
                mean_errors.append(np.nan)

        plt.bar(bin_centers, np.rad2deg(mean_errors), width=(self.distance_bins[1] - self.distance_bins[0]) * 0.9, color='salmon', edgecolor='black')
        plt.xlabel('Distance to Origin (units)')
        plt.ylabel('Mean Rotation Error (degrees)')
        plt.title('Rotation Error vs Distance')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def detection_rate_distance_bins(self):
        bins = self.distance_bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        total_counts = np.zeros(len(bins) - 1)
        valid_counts = np.zeros(len(bins) - 1)

        for index, row in self.data.iterrows():
            if pd.notna(row['tvec']):
                tvec = np.array([float(val) for val in row['tvec'].split(',')])
                distance = np.linalg.norm(tvec)
                bin_index = np.digitize(distance, bins) - 1
                if 0 <= bin_index < len(total_counts):
                    total_counts[bin_index] += 1
                    if 'n_valid' in row and pd.notna(row['n_valid']) and float(row['n_valid']) > 0:
                        valid_counts[bin_index] += 1

        detection_rates = np.divide(valid_counts, total_counts, out=np.zeros_like(valid_counts), where=total_counts != 0)

        plt.bar(bin_centers, detection_rates, width=(self.distance_bins[1] - self.distance_bins[0]) * 0.9, color='mediumseagreen', edgecolor='black')
        plt.xlabel('Distance to Origin (units)')
        plt.ylabel('Detection Rate')
        plt.title('Detection Rate vs Distance')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def detection_rate_angle_bins(self):
        bins = self.angle_bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        total_counts = np.zeros(len(bins) - 1)
        valid_counts = np.zeros(len(bins) - 1)

        for index, row in self.data.iterrows():
            if pd.notna(row['tvec']):
                tvec = np.array([float(val) for val in row['tvec'].split(',')])
                # Ângulo entre vetor e plano XY = ângulo entre vetor e seu componente no plano XY
                xy_norm = np.linalg.norm(tvec[:2])
                total_norm = np.linalg.norm(tvec)
                if total_norm > 0:
                    angle = np.arctan2(abs(tvec[2]), xy_norm)
                    bin_index = np.digitize(angle, bins) - 1
                    if 0 <= bin_index < len(total_counts):
                        total_counts[bin_index] += 1
                        if 'n_valid' in row and pd.notna(row['n_valid']) and float(row['n_valid']) > 0:
                            valid_counts[bin_index] += 1

        detection_rates = np.divide(valid_counts, total_counts, out=np.zeros_like(valid_counts), where=total_counts != 0)

        plt.bar(np.degrees(bin_centers), detection_rates, width=np.degrees(self.angle_bins[1] - self.angle_bins[0]) * 0.9,
                color='gold', edgecolor='black')
        plt.xlabel('Angle to XY Plane (degrees)')
        plt.ylabel('Detection Rate')
        plt.title('Detection Rate vs Angle to XY Plane')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def print_data_summary(self):
        """
        Prints a summary of the data including total entries, valid detections,
        and basic statistics on translation and rotation errors.
        """
        total_entries = len(self.data)
        valid_detections = self.data['n_valid'].dropna().astype(float).gt(0).sum()

        translation_errors = []
        rotation_errors = []

        for index, row in self.data.iterrows():
            if pd.notna(row['tvec']) and pd.notna(row['tvec_est']):
                tvec = np.array([float(val) for val in row['tvec'].split(',')])
                tvec_est = np.array([float(val) for val in row['tvec_est'].split(',')])
                translation_errors.append(np.linalg.norm(tvec - tvec_est))

            if pd.notna(row['rvec']) and pd.notna(row['rvec_est']):
                rvec = np.array([float(val) for val in row['rvec'].split(',')])
                rvec_est = np.array([float(val) for val in row['rvec_est'].split(',')])
                rvec_n = rvec / np.linalg.norm(rvec)
                rvec_est_n = rvec_est / np.linalg.norm(rvec_est)
                rotation_errors.append(np.arccos(np.clip(np.dot(rvec_n, rvec_est_n), -1.0, 1.0)))

        print(f"Total Entries: {total_entries}")
        print(f"Valid Detections: {valid_detections}")
        if translation_errors:
            print(f"Mean Translation Error: {np.mean(translation_errors):.4f} units")
            print(f"Median Translation Error: {np.median(translation_errors):.4f} units")
        if rotation_errors:
            print(f"Mean Rotation Error: {np.degrees(np.mean(rotation_errors)):.4f} degrees")
            print(f"Median Rotation Error: {np.degrees(np.median(rotation_errors)):.4f} degrees")

        # Print list with all angles x, y, z
        angles_x_real = []
        angles_y_real = []
        angles_z_real = []
        angles_x_est = []
        angles_y_est = []
        angles_z_est = []
        # Print angles to compare in list form
        # for index, row in self.data.iterrows():
        #     if pd.notna(row['rvec']) and pd.notna(row['rvec_est']):
        #         rvec = np.array([float(val) for val in row['rvec'].split(',')])
        #         rvec_est = np.array([float(val) for val in row['rvec_est'].split(',')])
        #         # Convert rotation vectors to rotation matrices
        #         R_real, _ = cv2.Rodrigues(rvec)
        #         R_est, _ = cv2.Rodrigues(rvec_est)
        #         # Extract Euler angles (in radians)
        #         angles_real = cv2.decomposeProjectionMatrix(np.hstack((R_real, np.zeros((3, 1)))))[6]
        #         angles_est = cv2.decomposeProjectionMatrix(np.hstack((R_est, np.zeros((3, 1)))))[6]
        #         angles_x_real.append(angles_real[0][0])
        #         angles_y_real.append(angles_real[1][0])
        #         angles_z_real.append(angles_real[2][0])
        #         angles_x_est.append(angles_est[0][0])
        #         angles_y_est.append(angles_est[1][0])
        #         angles_z_est.append(angles_est[2][0])
        # print("Real Angles X (radians):", angles_x_real)
        # print("Estimated Angles X (radians):", angles_x_est)
        # print("Real Angles Y (radians):", angles_y_real)
        # print("Estimated Angles Y (radians):", angles_y_est)
        # print("Real Angles Z (radians):", angles_z_real)
        # print("Estimated Angles Z (radians):", angles_z_est)
    
    def plot_rvec_comparison(self, index):
        """
        Plots the real and estimated rotation vectors (rvec and rvec_est) 
        for a given index to visualize angular differences.
        """
        row = self.data.iloc[index]

        if pd.isna(row['rvec']) or pd.isna(row['rvec_est']):
            print(f"No rotation data found for index {index}.")
            return

        rvec = np.array([float(val) for val in row['rvec'].split(',')])
        rvec_est = np.array([float(val) for val in row['rvec_est'].split(',')])

        # Normalize for fair direction comparison
        rvec_n = rvec / np.linalg.norm(rvec)
        rvec_est_n = rvec_est / np.linalg.norm(rvec_est)

        # Compute angular error in degrees
        dot_product = np.clip(np.dot(rvec_n, rvec_est_n), -1.0, 1.0)
        angle_error = np.degrees(np.arccos(dot_product))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the two vectors from the origin
        ax.quiver(0, 0, 0, rvec_n[0], rvec_n[1], rvec_n[2], 
                color='blue', label='Real rvec', linewidth=2)
        ax.quiver(0, 0, 0, rvec_est_n[0], rvec_est_n[1], rvec_est_n[2], 
                color='red', label='Estimated rvec', linewidth=2)

        # Axis setup
        max_range = 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f'Rotation Vector Comparison (Index {index})\nAngular Error = {angle_error:.2f}°')

        plt.show()

    def xy_translation_mean_error_bins(self):
        distances = []
        errors = []

        for index, row in self.data.iterrows():
            if pd.notna(row['tvec']) and pd.notna(row['tvec_est']):
                tvec = np.array([float(val) for val in row['tvec'].split(',')])
                tvec_est = np.array([float(val) for val in row['tvec_est'].split(',')])
                distance = np.linalg.norm(tvec)
                error = np.linalg.norm(tvec[:2] - tvec_est[:2])  # XY plane error
                distances.append(distance)
                errors.append(error)

        bins = self.distance_bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_errors = []

        for i in range(len(bins) - 1):
            in_bin = [(d, e) for d, e in zip(distances, errors) if bins[i] <= d < bins[i + 1]]
            if len(in_bin) > 0:
                mean_errors.append(np.mean([e for _, e in in_bin]))
            else:
                mean_errors.append(np.nan)

        plt.bar(bin_centers, mean_errors, width=(self.distance_bins[1] - self.distance_bins[0]) * 0.9, color='orchid', edgecolor='black')
        plt.xlabel('Distance to Origin (units)')
        plt.ylabel('Mean XY Translation Error (units)')
        plt.title('XY Translation Error vs Distance')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def z_translation_mean_error_bins(self):
        distances = []
        errors = []

        for index, row in self.data.iterrows():
            if pd.notna(row['tvec']) and pd.notna(row['tvec_est']):
                tvec = np.array([float(val) for val in row['tvec'].split(',')])
                tvec_est = np.array([float(val) for val in row['tvec_est'].split(',')])
                distance = np.linalg.norm(tvec)
                error = abs(tvec[2] - tvec_est[2])  # Z axis error
                distances.append(distance)
                errors.append(error)

        bins = self.distance_bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_errors = []

        for i in range(len(bins) - 1):
            in_bin = [(d, e) for d, e in zip(distances, errors) if bins[i] <= d < bins[i + 1]]
            if len(in_bin) > 0:
                mean_errors.append(np.mean([e for _, e in in_bin]))
            else:
                mean_errors.append(np.nan)

        plt.bar(bin_centers, mean_errors, width=(self.distance_bins[1] - self.distance_bins[0]) * 0.9, color='cyan', edgecolor='black')
        plt.xlabel('Distance to Origin (units)')
        plt.ylabel('Mean Z Translation Error (units)')
        plt.title('Z Translation Error vs Distance')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


if __name__ == "__main__":
    # Example usage
    csv_file = "C:\\Users\\eduar\\OneDrive\\Área de Trabalho\\bepe\\codes\\markers\\data\\d50\\results\\corners_2e_2e_1_with_poses.csv"
    graph_drawer = DrawGraphs(csv_file)
    
    # graph_drawer.plot_3d_poses()
    graph_drawer.print_data_summary()
    graph_drawer.plot_distance_difference_with_trend(abs_diff=True, show_r2=True)
    graph_drawer.plot_translation_errors_distance_bins()
    graph_drawer.plot_rotation_errors_distance_bins()
    graph_drawer.detection_rate_distance_bins()
    graph_drawer.detection_rate_angle_bins()
    graph_drawer.xy_translation_mean_error_bins()
    graph_drawer.z_translation_mean_error_bins()