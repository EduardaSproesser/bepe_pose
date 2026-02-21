import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DrawGraphs:
    def __init__(self, csv_file, estimation_type="single", marker_type="2e_2e", save_folder="results"):
        # Initialize any required variables or configurations
        # Get data from csv file rvecs, tvecs, n_valid, ids, rvec_est and tvec_est
        # The CSV produced by find_corners.py uses semicolons as separators and
        # contains commas inside vector fields (rvec/tvec/corners). Read with
        # sep=';' to avoid tokenization errors.
        self.data = pd.read_csv(csv_file, sep=';')
        self.estimation_type = estimation_type
        self.marker_type = marker_type
        
        # Criar pasta de resultados em formato: results/marker_type_estimation_type
        base_folder = save_folder
        self.save_folder = os.path.join(base_folder, f"{marker_type}_{estimation_type}")
        
        # Create save folder if it doesn't exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

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

    def _is_detection_valid(self, row):
        """Check if a detection is valid according to n_valid and special ID rules.
        
        Special rules:
        - 2e + single: only valid if marker id=1 is detected
        - 3e + single: only valid if marker id=3 is detected
        - Other configurations: valid if n_valid > 0
        
        Returns:
            bool: True if detection is valid, False otherwise
        """
        is_valid = False
        try:
            if 'n_valid' in row and pd.notna(row['n_valid']):
                n_valid = float(row['n_valid'])
                if n_valid > 0:
                    # Check for special ID requirements
                    if self.marker_type == '2e' and self.estimation_type == 'single':
                        # Only valid if id=1 is detected
                        if 'ids' in row and pd.notna(row['ids']):
                            ids_str = str(row['ids'])
                            # Parse ids - could be "1" or "1,2" or "[1]" etc
                            detected_ids = []
                            for char in ids_str:
                                if char.isdigit():
                                    detected_ids.append(int(char))
                            is_valid = 1 in detected_ids
                        else:
                            is_valid = False
                    elif self.marker_type == '3e' and self.estimation_type == 'single':
                        # Only valid if id=3 is detected
                        if 'ids' in row and pd.notna(row['ids']):
                            ids_str = str(row['ids'])
                            # Parse ids - could be "3" or "1,3" or "[3]" etc
                            detected_ids = []
                            for char in ids_str:
                                if char.isdigit():
                                    detected_ids.append(int(char))
                            is_valid = 3 in detected_ids
                        else:
                            is_valid = False
                    else:
                        # Default: valid if n_valid > 0
                        is_valid = True
        except:
            pass
        
        return is_valid

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
        filename = os.path.join(self.save_folder, f"3d_poses_{self.estimation_type}_{self.marker_type}.png")
        self.save_plot(fig, filename)
        plt.close()
    
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
            if not self._is_detection_valid(row):
                continue

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
        filename = os.path.join(self.save_folder, f"distance_difference_trend_{self.estimation_type}_{self.marker_type}.png")
        self.save_plot(fig, filename)
        plt.close()

        return m, b, r2
    
    def plot_angle_difference_with_trend(self, abs_diff=False, show_r2=True):
        # Similar to plot_distance_difference_with_trend, but for rotation angles
        angles = []
        deltas = []

        df = self.filter_df_by_pose_and_angle()  # por padrão aplica o filtro robusto

        for idx, row in df.iterrows():
            # somente entradas com detecção válida
            if not self._is_detection_valid(row):
                continue

            rvec = self._parse_vec(row.get('rvec', None))
            rvec_est = self._parse_vec(row.get('rvec_est', None))
            if rvec is None or rvec_est is None:
                continue

            # Cálculo do ângulo de diferença entre as rotações
            R_real, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            R_est, _ = cv2.Rodrigues(rvec_est.reshape(3, 1))
            R_diff = R_real.T @ R_est
            trace = np.trace(R_diff)
            cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
            angle_diff_rad = np.arccos(cos_angle)
            angle_diff_deg = np.degrees(angle_diff_rad)

            angle = np.linalg.norm(rvec)
            angles.append(angle)
            deltas.append(angle_diff_deg)

        if len(angles) == 0:
            print("[WARN] Nenhum dado válido encontrado para plotar.")
            return None, None, None

        x = np.array(angles)
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
        ax.scatter(x, y, s=30, alpha=0.7, label='Pontos (angle, Δangle)')
        # plot trendline sorted by x for a clean line
        order = np.argsort(x)
        ax.plot(x[order], y_fit[order], color='red', linewidth=2, label=f'Trend (y = {m:.3e} x + {b:.3e})')

        ax.set_xlabel('Rotation Angle (radians)')
        ax.set_ylabel('Absolute difference |angle - angle_est| (degrees)' if abs_diff else 'Signed difference (angle - angle_est) (degrees)')
        title = 'Difference of rotation angle vs Rotation angle'
        if show_r2:
            title += f' — R² = {r2:.4f}'
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        filename = os.path.join(self.save_folder, f"angle_difference_trend_{self.estimation_type}_{self.marker_type}.png")
        self.save_plot(fig, filename)
        plt.close()

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

        fig, ax = plt.subplots()
        ax.bar(bin_centers, mean_errors, width=(self.distance_bins[1] - self.distance_bins[0]) * 0.9, color='steelblue', edgecolor='black')
        ax.set_xlabel('Distance to Origin (units)')
        ax.set_ylabel('Mean Translation Error (units)')
        ax.set_title('Translation Error vs Distance')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        filename = os.path.join(self.save_folder, f"translation_errors_distance_bins_{self.estimation_type}_{self.marker_type}.png")
        self.save_plot(fig, filename)
        plt.close()

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
        fig, ax = plt.subplots()
        ax.bar(bin_centers, mean_errors, width=(self.distance_bins[1] - self.distance_bins[0]) * 0.9, color='coral', edgecolor='black')
        ax.set_xlabel('Distance to Origin (units)')
        ax.set_ylabel('Mean Rotation Error (degrees)')
        ax.set_title('Rotation Error vs Distance')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        filename = os.path.join(self.save_folder, f"rotation_errors_distance_bins_{self.estimation_type}_{self.marker_type}.png")
        self.save_plot(fig, filename)
        plt.close()
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
                    if self._is_detection_valid(row):
                        valid_counts[bin_index] += 1

        detection_rates = np.divide(valid_counts, total_counts, out=np.zeros_like(valid_counts), where=total_counts != 0)

        fig, ax = plt.subplots()
        ax.bar(bin_centers, detection_rates, width=(self.distance_bins[1] - self.distance_bins[0]) * 0.9, color='mediumseagreen', edgecolor='black')
        ax.set_xlabel('Distance to Origin (units)')
        ax.set_ylabel('Detection Rate')
        ax.set_title('Detection Rate vs Distance')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        filename = os.path.join(self.save_folder, f"detection_rate_distance_bins_{self.estimation_type}_{self.marker_type}.png")
        self.save_plot(fig, filename)
        plt.close()

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
                        if self._is_detection_valid(row):
                            valid_counts[bin_index] += 1

        detection_rates = np.divide(valid_counts, total_counts, out=np.zeros_like(valid_counts), where=total_counts != 0)

        fig, ax = plt.subplots()
        ax.bar(np.degrees(bin_centers), detection_rates, width=np.degrees(self.angle_bins[1] - self.angle_bins[0]) * 0.9,
                color='gold', edgecolor='black')
        ax.set_xlabel('Angle to XY Plane (degrees)')
        ax.set_ylabel('Detection Rate')
        ax.set_title('Detection Rate vs Angle to XY Plane')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        filename = os.path.join(self.save_folder, f"detection_rate_angle_bins_{self.estimation_type}_{self.marker_type}.png")
        self.save_plot(fig, filename)
        plt.close()

    def print_data_summary(self):
        """
        Prints a summary of the data including total entries, valid detections,
        and basic statistics on translation and rotation errors.
        """
        total_entries = len(self.data)
        
        # Contar detecções válidas usando o novo método
        valid_detections = sum(1 for _, row in self.data.iterrows() if self._is_detection_valid(row))

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

        plt.close()

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

        fig, ax = plt.subplots()
        ax.bar(bin_centers, mean_errors, width=(self.distance_bins[1] - self.distance_bins[0]) * 0.9, color='orchid', edgecolor='black')
        ax.set_xlabel('Distance to Origin (units)')
        ax.set_ylabel('Mean XY Translation Error (units)')
        ax.set_title('XY Translation Error vs Distance')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        filename = os.path.join(self.save_folder, f"xy_translation_mean_error_bins_{self.estimation_type}_{self.marker_type}.png")
        self.save_plot(fig, filename)
        plt.close()

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

        fig, ax = plt.subplots()
        ax.bar(bin_centers, mean_errors, width=(self.distance_bins[1] - self.distance_bins[0]) * 0.9, color='cyan', edgecolor='black')
        ax.set_xlabel('Distance to Origin (units)')
        ax.set_ylabel('Mean Z Translation Error (units)')
        ax.set_title('Z Translation Error vs Distance')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        filename = os.path.join(self.save_folder, f"z_translation_mean_error_bins_{self.estimation_type}_{self.marker_type}.png")
        self.save_plot(fig, filename)
        plt.close()

    def save_plot(self, fig, filename):
        """
        Saves the current plot to a file.
        """
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        fig.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"Plot saved to {filename}")

    def generate_error_table_by_bins(self):
        """
        Gera uma tabela com erros médios e taxa de detecção para cada intervalo
        de distância e ângulo.
        
        Retorna um DataFrame com as seguintes colunas:
        - Bin Type (Distance/Angle)
        - Bin Range
        - Mean Translation Error
        - Mean Rotation Error (degrees)
        - Detection Rate
        - Sample Count
        """
        df = self.filter_df_by_pose_and_angle()
        results = []
        
        # ========== DISTANCE BINS ==========
        bins_dist = self.distance_bins
        for i in range(len(bins_dist) - 1):
            bin_range = f"{bins_dist[i]:.0f}-{bins_dist[i+1]:.0f}"
            
            # Coletar dados neste bin
            trans_errors = []
            rot_errors = []
            total_count = 0
            valid_count = 0
            
            for idx, row in df.iterrows():
                tvec = self._parse_vec(row.get('tvec'))
                if tvec is None:
                    continue
                    
                distance = np.linalg.norm(tvec)
                if bins_dist[i] <= distance < bins_dist[i+1]:
                    total_count += 1
                    
                    # Detecção válida
                    if self._is_detection_valid(row):
                        valid_count += 1
                    
                    # Erro de translação
                    tvec_est = self._parse_vec(row.get('tvec_est'))
                    if tvec_est is not None:
                        trans_error = np.linalg.norm(tvec - tvec_est)
                        trans_errors.append(trans_error)
                    
                    # Erro de rotação
                    rvec = self._parse_vec(row.get('rvec'))
                    rvec_est = self._parse_vec(row.get('rvec_est'))
                    if rvec is not None and rvec_est is not None:
                        R_real, _ = cv2.Rodrigues(rvec.reshape(3, 1))
                        R_est, _ = cv2.Rodrigues(rvec_est.reshape(3, 1))
                        R_diff = R_real.T @ R_est
                        trace = np.trace(R_diff)
                        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
                        rot_error_rad = np.arccos(cos_angle)
                        rot_errors.append(np.degrees(rot_error_rad))
            
            mean_trans_error = np.mean(trans_errors) if len(trans_errors) > 0 else np.nan
            mean_rot_error = np.mean(rot_errors) if len(rot_errors) > 0 else np.nan
            detection_rate = valid_count / total_count if total_count > 0 else np.nan
            
            results.append({
                'Bin_Type': 'Distance',
                'Bin_Range': bin_range,
                'Mean_Translation_Error': mean_trans_error,
                'Mean_Rotation_Error_deg': mean_rot_error,
                'Detection_Rate': detection_rate,
                'Sample_Count': total_count
            })
        
        # ========== ANGLE BINS ==========
        bins_angle = self.angle_bins
        for i in range(len(bins_angle) - 1):
            bin_range = f"{np.degrees(bins_angle[i]):.0f}-{np.degrees(bins_angle[i+1]):.0f}"
            
            # Coletar dados neste bin
            trans_errors = []
            rot_errors = []
            total_count = 0
            valid_count = 0
            
            for idx, row in df.iterrows():
                tvec = self._parse_vec(row.get('tvec'))
                if tvec is None:
                    continue
                
                # Calcular ângulo
                xy_norm = np.linalg.norm(tvec[:2])
                total_norm = np.linalg.norm(tvec)
                if total_norm == 0:
                    continue
                    
                angle = np.arctan2(abs(tvec[2]), xy_norm)
                if bins_angle[i] <= angle < bins_angle[i+1]:
                    total_count += 1
                    
                    # Detecção válida
                    if self._is_detection_valid(row):
                        valid_count += 1
                    
                    # Erro de translação
                    tvec_est = self._parse_vec(row.get('tvec_est'))
                    if tvec_est is not None:
                        trans_error = np.linalg.norm(tvec - tvec_est)
                        trans_errors.append(trans_error)
                    
                    # Erro de rotação
                    rvec = self._parse_vec(row.get('rvec'))
                    rvec_est = self._parse_vec(row.get('rvec_est'))
                    if rvec is not None and rvec_est is not None:
                        R_real, _ = cv2.Rodrigues(rvec.reshape(3, 1))
                        R_est, _ = cv2.Rodrigues(rvec_est.reshape(3, 1))
                        R_diff = R_real.T @ R_est
                        trace = np.trace(R_diff)
                        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
                        rot_error_rad = np.arccos(cos_angle)
                        rot_errors.append(np.degrees(rot_error_rad))
            
            mean_trans_error = np.mean(trans_errors) if len(trans_errors) > 0 else np.nan
            mean_rot_error = np.mean(rot_errors) if len(rot_errors) > 0 else np.nan
            detection_rate = valid_count / total_count if total_count > 0 else np.nan
            
            results.append({
                'Bin_Type': 'Angle',
                'Bin_Range': bin_range,
                'Mean_Translation_Error': mean_trans_error,
                'Mean_Rotation_Error_deg': mean_rot_error,
                'Detection_Rate': detection_rate,
                'Sample_Count': total_count
            })
        
        return pd.DataFrame(results)

    def plot_detection_heatmap_2d(self, axis='xy'):
        """
        Cria um gráfico em 2D mostrando quais posições foram detectadas (verde)
        e quais não foram detectadas (vermelho).
        
        Args:
            axis: 'xy', 'xz' ou 'yz' - plano a ser visualizado
        """
        detected_x = []
        detected_y = []
        not_detected_x = []
        not_detected_y = []
        
        df = self.filter_df_by_pose_and_angle()
        
        axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        if axis not in axis_map:
            print(f"Eixo inválido. Use: {list(axis_map.keys())}")
            return
        
        ax1, ax2 = axis_map[axis]
        axis_names = ['X', 'Y', 'Z']
        
        for idx, row in df.iterrows():
            tvec = self._parse_vec(row.get('tvec'))
            if tvec is None:
                continue
            
            if self._is_detection_valid(row):
                detected_x.append(tvec[ax1])
                detected_y.append(tvec[ax2])
            else:
                not_detected_x.append(tvec[ax1])
                not_detected_y.append(tvec[ax2])
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Pontos não detectados em vermelho
        if len(not_detected_x) > 0:
            ax.scatter(not_detected_x, not_detected_y, c='red', s=100, alpha=0.6, 
                      label='Não Detectado', marker='x', linewidths=2)
        
        # Pontos detectados em verde
        if len(detected_x) > 0:
            ax.scatter(detected_x, detected_y, c='green', s=100, alpha=0.6, 
                      label='Detectado', marker='o')
        
        ax.set_xlabel(f'{axis_names[ax1]} (units)', fontsize=12)
        ax.set_ylabel(f'{axis_names[ax2]} (units)', fontsize=12)
        ax.set_title(f'Mapa de Detecções - Plano {axis.upper()}\n{self.marker_type} - {self.estimation_type}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        plt.tight_layout()
        
        filename = os.path.join(self.save_folder, f"detection_heatmap_2d_{axis}_{self.estimation_type}_{self.marker_type}.png")
        self.save_plot(fig, filename)
        plt.close()

    def plot_detection_heatmap_3d(self):
        """
        Cria uma visualização onde cada posição é um quadradinho separado.
        Verde = detectado, Vermelho = não detectado.
        Cada quadradinho representa uma posição única.
        """
        df = self.filter_df_by_pose_and_angle()
        
        # Coletar todas as posições
        positions = []
        for idx, row in df.iterrows():
            tvec = self._parse_vec(row.get('tvec'))
            if tvec is None:
                continue
            
            distance = np.linalg.norm(tvec)
            xy_norm = np.linalg.norm(tvec[:2])
            total_norm = np.linalg.norm(tvec)
            
            if total_norm == 0:
                continue
            
            angle = np.arctan2(abs(tvec[2]), xy_norm)
            
            # Verificar se foi detectado
            is_valid = self._is_detection_valid(row)
            
            positions.append({
                'distance': distance,
                'angle': angle,
                'detected': is_valid,
                'idx': idx
            })
        
        if len(positions) == 0:
            print("[WARN] Nenhuma posição foi encontrada.")
            return
        
        # Ordenar por distância e depois por ângulo
        positions_sorted = sorted(positions, key=lambda x: (x['distance'], x['angle']))
        
        # Criar matriz de cores: 1 para cada posição
        n_positions = len(positions_sorted)
        
        # Decidir layout (aproximadamente quadrado)
        n_cols = int(np.ceil(np.sqrt(n_positions)))
        n_rows = int(np.ceil(n_positions / n_cols))
        
        # Criar matriz
        color_matrix = np.zeros((n_rows, n_cols, 3))
        labels_matrix = np.full((n_rows, n_cols), '', dtype=object)
        
        # Preencher matriz
        for idx, pos in enumerate(positions_sorted):
            row = idx // n_cols
            col = idx % n_cols
            
            if pos['detected']:
                # Verde
                color_matrix[row, col] = [0.0, 0.8, 0.0]
                label = "✓"
            else:
                # Vermelho
                color_matrix[row, col] = [0.9, 0.0, 0.0]
                label = "✗"
            
            labels_matrix[row, col] = label
        
        # Criar figura com células bem visíveis
        fig, ax = plt.subplots(figsize=(20, 16))
        
        im = ax.imshow(color_matrix, aspect='auto')
        
        # Remover ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Adicionar labels nas células
        for row in range(n_rows):
            for col in range(n_cols):
                idx = row * n_cols + col
                if idx < n_positions:
                    pos = positions_sorted[idx]
                    label = labels_matrix[row, col]
                    text_color = 'white'
                    
                    ax.text(col, row, label, ha="center", va="center",
                           color=text_color, fontsize=6, fontweight='bold')
        
        # Título
        detected_count = sum(1 for p in positions_sorted if p['detected'])
        not_detected_count = len(positions_sorted) - detected_count
        detection_rate = detected_count / len(positions_sorted) * 100 if len(positions_sorted) > 0 else 0
        
        ax.set_title(
            f'Mapa de Detecções - Cada quadradinho é uma posição\n'
            f'{self.marker_type} - {self.estimation_type}\n'
            f'Total: {n_positions} posições | Detectadas: {detected_count} ({detection_rate:.1f}%) | Não detectadas: {not_detected_count}',
            fontsize=14, fontweight='bold', pad=20
        )
        
        # Adicionar legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0.0, 0.8, 0.0], label='Detectado (✓)'),
            Patch(facecolor=[0.9, 0.0, 0.0], label='Não Detectado (✗)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=12)
        
        plt.tight_layout()
        
        filename = os.path.join(self.save_folder, f"detection_heatmap_individual_{self.estimation_type}_{self.marker_type}.png")
        self.save_plot(fig, filename)
        plt.close()

def generate_complete_error_table(base_path, marker_types, estimation_types, output_file="complete_error_table.csv"):
    """
    Gera uma tabela completa com erros médios e taxa de detecção para todos os
    marker_types e estimation_types.
    
    Args:
        base_path: Caminho base onde estão os arquivos CSV
        marker_types: Lista de tipos de marcadores (ex: ["1p", "2e", "3e", "2v", "3v"])
        estimation_types: Lista de tipos de estimação (ex: ["single", "multi_iterative", "multi_mean"])
        output_file: Nome do arquivo de saída
    
    Retorna:
        DataFrame com todos os resultados
    """
    all_results = []
    
    for marker_type in marker_types:
        for estimation_type in estimation_types:
            print(f"Processando {marker_type} - {estimation_type}...")
            
            try:
                # Construir caminhos dos arquivos CSV
                csv_file1 = os.path.join(base_path, f"corners_{marker_type}_{marker_type}_1_with_poses_{estimation_type}.csv")
                csv_file2 = os.path.join(base_path, f"corners_{marker_type}_{marker_type}_2_with_poses_{estimation_type}.csv")
                csv_file3 = os.path.join(base_path, f"corners_{marker_type}_{marker_type}_3_with_poses_{estimation_type}.csv")
                
                # Verificar se os arquivos existem
                if not all(os.path.exists(f) for f in [csv_file1, csv_file2, csv_file3]):
                    print(f"  Arquivos não encontrados para {marker_type} - {estimation_type}, pulando...")
                    continue
                
                # Combinar CSVs
                df1 = pd.read_csv(csv_file1, sep=';')
                df2 = pd.read_csv(csv_file2, sep=';')
                df3 = pd.read_csv(csv_file3, sep=';')
                combined_df = pd.concat([df1, df2, df3], ignore_index=True)
                
                # Salvar CSV combinado temporário
                temp_csv = f"temp_combined_{marker_type}_{estimation_type}.csv"
                combined_df.to_csv(temp_csv, sep=';', index=False)
                
                # Criar instância e gerar tabela
                graph_drawer = DrawGraphs(temp_csv, estimation_type=estimation_type, 
                                         marker_type=marker_type, save_folder="results")
                bin_table = graph_drawer.generate_error_table_by_bins()
                
                # Adicionar identificadores
                bin_table['Marker_Type'] = marker_type
                bin_table['Estimation_Type'] = estimation_type
                
                all_results.append(bin_table)
                
                # Remover arquivo temporário
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
                    
            except Exception as e:
                print(f"  Erro ao processar {marker_type} - {estimation_type}: {str(e)}")
                continue
    
    if len(all_results) == 0:
        print("Nenhum resultado foi gerado!")
        return None
    
    # Combinar todos os resultados
    final_table = pd.concat(all_results, ignore_index=True)
    
    # Reordenar colunas para melhor legibilidade
    columns_order = ['Marker_Type', 'Estimation_Type', 'Bin_Type', 'Bin_Range', 
                     'Mean_Translation_Error', 'Mean_Rotation_Error_deg', 
                     'Detection_Rate', 'Sample_Count']
    final_table = final_table[columns_order]
    
    # Salvar em arquivo
    final_table.to_csv(output_file, index=False)
    print(f"\nTabela completa salva em: {output_file}")
    print(f"Total de linhas: {len(final_table)}")
    
    return final_table


if __name__ == "__main__":
    # ========== CONFIGURAÇÃO ==========
    base_path = r"C:\Users\eduar\OneDrive\Área de Trabalho\bepe\codes\markers\data\d50\results"
    results_folder = "results"
    
    # Todas as combinações
    marker_types = ["1p", "2e", "3e", "2v", "3v"]
    estimation_types = ["single", "multi_iterative", "multi_mean"]
    
    print("="*80)
    print("GERANDO GRÁFICOS DE ANÁLISE PARA TODAS AS COMBINAÇÕES")
    print("="*80)
    
    total_processed = 0
    total_skipped = 0
    
    for marker_type in marker_types:
        for estimation_type in estimation_types:
            print(f"\n{'─'*80}")
            print(f"Processando {marker_type} - {estimation_type}...")
            print(f"{'─'*80}")
            
            try:
                # Construir caminhos dos arquivos CSV
                csv_file1 = os.path.join(base_path, f"corners_{marker_type}_{marker_type}_1_with_poses_{estimation_type}.csv")
                csv_file2 = os.path.join(base_path, f"corners_{marker_type}_{marker_type}_2_with_poses_{estimation_type}.csv")
                csv_file3 = os.path.join(base_path, f"corners_{marker_type}_{marker_type}_3_with_poses_{estimation_type}.csv")
                
                # Verificar se os arquivos existem
                if not all(os.path.exists(f) for f in [csv_file1, csv_file2, csv_file3]):
                    print(f"⚠️  Arquivos não encontrados, pulando...")
                    total_skipped += 1
                    continue
                
                print(f"✓ Arquivos CSV encontrados")
                
                # Combinar CSVs
                df1 = pd.read_csv(csv_file1, sep=';')
                df2 = pd.read_csv(csv_file2, sep=';')
                df3 = pd.read_csv(csv_file3, sep=';')
                combined_df = pd.concat([df1, df2, df3], ignore_index=True)
                
                # Salvar CSV combinado temporário
                temp_csv = f"temp_combined_{marker_type}_{estimation_type}.csv"
                combined_df.to_csv(temp_csv, sep=';', index=False)
                
                print(f"✓ CSV combinado carregado ({len(combined_df)} linhas)")
                
                # Criar instância
                graph_drawer = DrawGraphs(temp_csv, estimation_type=estimation_type, 
                                         marker_type=marker_type, save_folder=results_folder)
                
                print(f"\n📊 Gerando gráficos...\n")
                
                print(f"  ✓ Resumo dos dados...")
                graph_drawer.print_data_summary()
                
                print(f"  ✓ Diferença de distância com tendência...")
                graph_drawer.plot_distance_difference_with_trend(abs_diff=True, show_r2=True)
                
                print(f"  ✓ Diferença de ângulo com tendência...")
                graph_drawer.plot_angle_difference_with_trend(abs_diff=True, show_r2=True)
                
                print(f"  ✓ Erros de translação por bins de distância...")
                graph_drawer.plot_translation_errors_distance_bins()
                
                print(f"  ✓ Erros de rotação por bins de distância...")
                graph_drawer.plot_rotation_errors_distance_bins()
                
                print(f"  ✓ Taxa de detecção por bins de distância...")
                graph_drawer.detection_rate_distance_bins()
                
                print(f"  ✓ Taxa de detecção por bins de ângulo...")
                graph_drawer.detection_rate_angle_bins()
                
                print(f"  ✓ Erros de translação XY por bins de distância...")
                graph_drawer.xy_translation_mean_error_bins()
                
                print(f"  ✓ Erros de translação Z por bins de distância...")
                graph_drawer.z_translation_mean_error_bins()
                
                print(f"  ✓ Mapa de detecções individual...")
                graph_drawer.plot_detection_heatmap_3d()
                
                # Remover arquivo temporário
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
                
                print(f"\n✅ {marker_type} - {estimation_type} concluído com sucesso!")
                print(f"📁 Resultados: {results_folder}/{marker_type}_{estimation_type}/")
                total_processed += 1
            
            except Exception as e:
                print(f"❌ Erro ao processar {marker_type} - {estimation_type}: {str(e)}")
                total_skipped += 1
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n" + "="*80)
    print(f"✅ PROCESSAMENTO CONCLUÍDO!")
    print(f"  • Combinações processadas: {total_processed}")
    print(f"  • Combinações puladas: {total_skipped}")
    print(f"  • Total: {total_processed + total_skipped}")
    print(f"📁 Todos os resultados estão em: {results_folder}/")
    print(f"="*80)