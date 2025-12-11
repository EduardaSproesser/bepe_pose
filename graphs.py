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

    # ----------------- Helpers for parsing and outlier filtering -----------------
    def _parse_vec(self, s):
        """Parse a string like "x,y,z" or an iterable into a numpy array of shape (3,).
        Returns None when parsing fails.
        """
        if pd.isna(s):
            return None
        if isinstance(s, (list, tuple, np.ndarray)):
            arr = np.array(s, dtype=float)
            return arr if arr.size == 3 else None
        try:
            parts = [p for p in str(s).split(',') if p.strip() != '']
            if len(parts) != 3:
                return None
            return np.array([float(p) for p in parts], dtype=float)
        except Exception:
            return None

    def filter_df_by_pose_and_angle(self, thresh=3.5, features=('tvec', 'rvec')):
        """Return a filtered copy of self.data removing pose and angle outliers.

        Strategy (robust): compute modified z-score using MAD for selected pose
        features (tvec components, tvec norm, rvec components, rvec norm) and
        additionally the viewing angle of tvec relative to the XY plane. A row
        is removed when any feature's modified z-score exceeds `thresh`.

        Returns a DataFrame with only rows that have parseable `tvec` and `rvec`
        and that are not flagged as outliers.
        """
        df = self.data.copy()

        parsed_t = []
        parsed_r = []
        valid_idx = []

        # parse all rows first
        for idx, row in df.iterrows():
            t = self._parse_vec(row.get('tvec', None))
            r = self._parse_vec(row.get('rvec', None))
            if t is None or r is None:
                parsed_t.append(None)
                parsed_r.append(None)
                continue
            parsed_t.append(t)
            parsed_r.append(r)
            valid_idx.append(idx)

        if len(valid_idx) == 0:
            return df.iloc[[]]  # empty

        # ... (código anterior) ...

        tvecs = np.vstack([parsed_t[i] for i in range(len(parsed_t)) if parsed_t[i] is not None])
        rvecs = np.vstack([parsed_r[i] for i in range(len(parsed_r)) if parsed_r[i] is not None])
        
        # --- Cálculo de Features de Pose (Continuação) ---
        t_norm = np.linalg.norm(tvecs, axis=1)
        r_norm = np.linalg.norm(rvecs, axis=1)

        # Adicionar viewing angle (angle between tvec and its projection on XY plane)
        xy_norm = np.linalg.norm(tvecs[:, :2], axis=1)
        angles = np.arctan2(np.abs(tvecs[:, 2]), xy_norm)

        # --------------------------------------------------------------------------
        # NOVO: Adicionar uma feature de Rotação para detectar os 'flips' 180-degree
        # Vamos usar a variação do ângulo de rotação (magnitude do rvec)
        # --------------------------------------------------------------------------
        
        # O método MAD é bom, mas é mais vulnerável para rvecs, que são cíclicos.
        # Continuaremos usando rvecs, r_norm e tvecs, t_norm, e angles
        
        # Build feature matrix aligned with valid rows
        # Removemos r_norm por enquanto, pois é menos informativo para flips
        feats = np.hstack([tvecs, t_norm.reshape(-1, 1), rvecs, angles.reshape(-1, 1)]) # 3 T, 1 T_norm, 3 R, 1 Angle = 8 features
        
        # --------------------------------------------------------------------------
        # O resto do filtro (MAD Z-score) permanece
        # --------------------------------------------------------------------------

        # Robust modified z-score per column (using MAD)
        medians = np.median(feats, axis=0)
        mad = np.median(np.abs(feats - medians), axis=0)

        # ... (código para calcular modified_z) ...
        # ... (código para calcular outlier_mask_valid_rows) ...
        # ... (código para retornar df[keep_mask]) ...

        # Avoid division by zero: fallback to std if mad == 0, else fall back to no-scaling
        stds = np.std(feats, axis=0)

        modified_z = np.zeros_like(feats)
        for j in range(feats.shape[1]):
            if mad[j] > 0:
                modified_z[:, j] = 0.6745 * (feats[:, j] - medians[j]) / mad[j]
            elif stds[j] > 0:
                modified_z[:, j] = (feats[:, j] - np.mean(feats[:, j])) / stds[j]
            else:
                modified_z[:, j] = 0.0

        # Mark outliers where any feature exceeds threshold
        outlier_mask_valid_rows = np.any(np.abs(modified_z) > thresh, axis=1)

        # Build final boolean keep mask aligned with df.index
        keep_mask = np.zeros(len(df), dtype=bool)
        feat_idx = 0
        for pos, parsed in enumerate(parsed_t):
            if parsed is None:
                keep_mask[pos] = False
            else:
                keep_mask[pos] = not outlier_mask_valid_rows[feat_idx]
                feat_idx += 1

        return df[keep_mask].reset_index(drop=True)


    def plot_3d_poses(self, filter_outliers=True):
        # Plot real and estimates poses in 3D grid (only points)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        df = self.filter_df_by_pose_and_angle() if filter_outliers else self.data

        for index, row in df.iterrows():
            # Real poses
            rvec_str = row['rvec']
            tvec_str = row['tvec']
            if pd.notna(rvec_str) and pd.notna(tvec_str):
                rvec = self._parse_vec(rvec_str)
                tvec = self._parse_vec(tvec_str)
                ax.scatter(tvec[0], tvec[1], tvec[2], c='b', marker='o', label='Real' if index == 0 else "")
            
            # Estimated poses
            rvec_est_str = row['rvec_est']
            tvec_est_str = row['tvec_est']
            if pd.notna(rvec_est_str) and pd.notna(tvec_est_str):
                rvec_est = self._parse_vec(rvec_est_str)
                tvec_est = self._parse_vec(tvec_est_str)
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

        df = self.filter_df_by_pose_and_angle()  # por padrão aplica o filtro robusto

        for idx, row in df.iterrows():
            # somente entradas com detecção válida
            try:
                if 'n_valid' in row and (pd.isna(row['n_valid']) or float(row['n_valid']) <= 0):
                    continue
            except Exception:
                # se n_valid não for numérico, tenta converter; caso falhe, assume válido
                pass

            tvec = self._parse_vec(row.get('tvec', None))
            tvec_est = self._parse_vec(row.get('tvec_est', None))
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
        df = self.filter_df_by_pose_and_angle()

        for index, row in df.iterrows():
            if pd.notna(row.get('tvec')) and pd.notna(row.get('tvec_est')):
                tvec = self._parse_vec(row.get('tvec'))
                tvec_est = self._parse_vec(row.get('tvec_est'))
                if tvec is None or tvec_est is None:
                    continue
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
        df = self.filter_df_by_pose_and_angle()

        for index, row in df.iterrows():
            if pd.notna(row.get('rvec')) and pd.notna(row.get('rvec_est')) and pd.notna(row.get('tvec')):
                rvec = self._parse_vec(row.get('rvec'))
                rvec_est = self._parse_vec(row.get('rvec_est'))
                tvec = self._parse_vec(row.get('tvec'))
                if rvec is None or rvec_est is None or tvec is None:
                    continue
                
                distance = np.linalg.norm(tvec)
                
                # --- NOVO CÁLCULO DE ERRO DE ROTAÇÃO (ROBUSTO) ---
                # 1. Converter rvecs para Matrizes de Rotação (R)
                R_real, _ = cv2.Rodrigues(rvec.reshape(3, 1))
                R_est, _ = cv2.Rodrigues(rvec_est.reshape(3, 1))
                
                # 2. Calcular a Matriz de Rotação de Diferença: R_diff = R_real.T @ R_est
                R_diff = R_real.T @ R_est
                
                # 3. Calcular o ângulo de rotação (erro) a partir do R_diff (fórmula do Ângulo-Eixo)
                trace = np.trace(R_diff)
                
                # np.clip previne erros de ponto flutuante fora da faixa [-1, 1]
                cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
                error = np.arccos(cos_angle)
                
                # --- FIM DO NOVO CÁLCULO ---
                # Change to degrees
                distances.append(distance)
                error = np.rad2deg(error)
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
        plt.bar(bin_centers, mean_errors, width=(self.distance_bins[1] - self.distance_bins[0]) * 0.9, color='coral', edgecolor='black')
        plt.xlabel('Distance to Origin (units)')
        plt.ylabel('Mean Rotation Error (degrees)')
        plt.title('Rotation Error vs Distance')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        # ... (Restante do código da função plot_rotation_errors_distance_bins)

    def detection_rate_distance_bins(self):
        bins = self.distance_bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        total_counts = np.zeros(len(bins) - 1)
        valid_counts = np.zeros(len(bins) - 1)

        df = self.filter_df_by_pose_and_angle()

        for index, row in df.iterrows():
            if pd.notna(row.get('tvec')):
                tvec = self._parse_vec(row.get('tvec'))
                if tvec is None:
                    continue
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

        df = self.filter_df_by_pose_and_angle()

        for index, row in df.iterrows():
            if pd.notna(row.get('tvec')):
                tvec = self._parse_vec(row.get('tvec'))
                if tvec is None:
                    continue
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
            # ... (Cálculo de translation_errors - OMITIDO, pois está correto) ...
            if pd.notna(row['tvec']) and pd.notna(row['tvec_est']):
                tvec = self._parse_vec(row.get('tvec'))
                tvec_est = self._parse_vec(row.get('tvec_est'))
                if tvec is not None and tvec_est is not None:
                    translation_errors.append(np.linalg.norm(tvec - tvec_est))

            if pd.notna(row['rvec']) and pd.notna(row['rvec_est']):
                rvec = self._parse_vec(row.get('rvec'))
                rvec_est = self._parse_vec(row.get('rvec_est'))
                if rvec is None or rvec_est is None:
                    continue
                
                # --- NOVO CÁLCULO DE ERRO DE ROTAÇÃO (ROBUSTO) ---
                # 1. Converter rvecs para Matrizes de Rotação (R)
                R_real, _ = cv2.Rodrigues(rvec.reshape(3, 1))
                R_est, _ = cv2.Rodrigues(rvec_est.reshape(3, 1))
                
                # 2. Calcular a Matriz de Rotação de Diferença: R_diff = R_real.T @ R_est
                R_diff = R_real.T @ R_est
                
                # 3. Calcular o ângulo de rotação (erro) a partir do R_diff
                trace = np.trace(R_diff)
                cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                
                rotation_errors.append(angle_rad)
                # --- FIM DO NOVO CÁLCULO ---

        print(f"Total Entries: {total_entries}")
        print(f"Valid Detections: {valid_detections}")
        print(f"Translation Error - Mean: {np.mean(translation_errors):.4f}, Std: {np.std(translation_errors):.4f}")
        print(f"Translation Error - Median: {np.median(translation_errors):.4f}, 90th Percentile: {np.percentile(translation_errors, 90):.4f}")
        print(f"Rotation Error (deg) - Mean: {np.mean(np.degrees(rotation_errors)):.4f}, Std: {np.std(np.degrees(rotation_errors)):.4f}")
        print(f"Rotation Error (deg) - Median: {np.median(np.degrees(rotation_errors)):.4f}, 90th Percentile: {np.percentile(np.degrees(rotation_errors), 90):.4f}")
        # ... (Restante da impressão de resultados) ...
    
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

        df = self.filter_df_by_pose_and_angle()

        for index, row in df.iterrows():
            if pd.notna(row.get('tvec')) and pd.notna(row.get('tvec_est')):
                tvec = self._parse_vec(row.get('tvec'))
                tvec_est = self._parse_vec(row.get('tvec_est'))
                if tvec is None or tvec_est is None:
                    continue
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

        df = self.filter_df_by_pose_and_angle()

        for index, row in df.iterrows():
            if pd.notna(row.get('tvec')) and pd.notna(row.get('tvec_est')):
                tvec = self._parse_vec(row.get('tvec'))
                tvec_est = self._parse_vec(row.get('tvec_est'))
                if tvec is None or tvec_est is None:
                    continue
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
    csv_file = "C:\\Users\\eduar\\OneDrive\\Área de Trabalho\\bepe\\codes\\markers\\data\\d50\\results\\corners_2e_2e_2_with_poses.csv"
    graph_drawer = DrawGraphs(csv_file)
    
    graph_drawer.plot_3d_poses()
    graph_drawer.print_data_summary()
    graph_drawer.plot_distance_difference_with_trend(abs_diff=True, show_r2=True)
    graph_drawer.plot_translation_errors_distance_bins()
    graph_drawer.plot_rotation_errors_distance_bins()
    graph_drawer.detection_rate_distance_bins()
    graph_drawer.detection_rate_angle_bins()
    graph_drawer.xy_translation_mean_error_bins()
    graph_drawer.z_translation_mean_error_bins()

    for i in range(len(graph_drawer.data)):
        graph_drawer.plot_rvec_comparison(i)