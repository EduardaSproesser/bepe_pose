import cv2
import numpy as np


def detect_and_preview(image_path, marker_length, camera_matrix, dist_coeffs):
    """
    Detect ArUco markers in an image and display them with IDs and axes.
    
    Params:
        image_path (str): Path to the image file.
        marker_length (float): Marker size in meters.
        camera_matrix (np.ndarray): 3×3 camera matrix from calibration.
        dist_coeffs (np.ndarray): Distortion coefficients.
    """

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Select dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Parameters
    try:
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    except:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Draw detected markers
    output = image.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(output, corners, ids)

        # For each marker: estimate pose + draw axis
        for corner, marker_id in zip(corners, ids):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner, marker_length, camera_matrix, dist_coeffs
            )
            rvec = rvec[0][0]
            tvec = tvec[0][0]

            # Draw coordinate axes (length = marker_length)
            cv2.drawFrameAxes(output, camera_matrix, dist_coeffs, rvec, tvec, marker_length)

    # Show
    cv2.imshow("ArUco Detection Preview", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ids


# ------------------ Example usage ------------------

if __name__ == "__main__":

    image_path = "test_images/0234.jpg"  # change to your image path

    # Example camera calibration (replace with your calibration!)
    camera_matrix = np.array([
         [ 2.83125434e+03, -1.69530489e+00,  1.65646731e+03],
 [ 0.00000000e+00,  2.83164603e+03,  1.23673955e+03],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ], dtype=np.float32)

    dist_coeffs = np.array([[-4.96032054e-01,  2.91734880e-01,  3.83732201e-04, -1.39581662e-03]], dtype=np.float32)  # or use your calibrated distortion

    marker_length = 0.02  # 20 mm marker

    detect_and_preview(image_path, marker_length, camera_matrix, dist_coeffs)
