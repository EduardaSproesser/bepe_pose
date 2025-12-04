import numpy as np

def rigid_transform_3D(A, B):
    """
    Compute R, t such that B = R*A + t
    A and B are Nx3 matrices of corresponding 3D points.
    Uses SVD-based Umeyama rigid alignment (no scaling).
    """
    assert A.shape == B.shape

    # Centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center points
    AA = A - centroid_A
    BB = B - centroid_B

    # Correlation matrix
    H = AA.T @ BB

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflections)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = centroid_B - R @ centroid_A

    return R, t


def compute_marker_pose(marker_corners_smf, marker_size):
    """
    marker_corners_smf: (4,3) array with the corners in SMF coordinates
    marker_size: size of the marker (float)
    Returns R, t such that X_smf = R * X_marker + t
    """

    obj_points = np.array([
        [-marker_size/2,  marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)

    R, t = rigid_transform_3D(obj_points, marker_corners_smf)
    return R, t


if __name__ == "__main__":
    # EXEMPLO DE USO
    marker_type = "3v"
    marker_size_dict = {"1p": 8.2, "2e": 5.3, "3e": 4.35, "2v": 4.2, "3v": 4.2}
    marker_size = marker_size_dict.get(marker_type)

    match marker_type:
        case "1p":
            marker_points_dict = {
                5: np.array([
                    [-4.1, 4.1, 3.09],
                    [4.1, 4.1, 3.09],
                    [4.1, -4.1, 3.09],
                    [-4.1, -4.1, 3.09]
                ], dtype=np.float32)
            }
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
    for marker_id, corners_smf in marker_points_dict.items():
        R, t = compute_marker_pose(corners_smf, marker_size)
        print(f"Marker ID: {marker_id}")

        # Print tranformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        print("Transformation Matrix:\n", T)