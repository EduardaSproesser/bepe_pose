# Code to find aruco corners in an image using OpenCV
import cv2
import numpy as np
from re import match
import pandas as pd
import os


def imread_with_fallback(path):
    """Attempt to read an image with cv2.imread, and if that returns None,
    try reading file bytes and decode with cv2.imdecode. This helps on Windows
    when cv2.imread fails on paths with non-ASCII characters.
    """
    # Prefer reading raw bytes first and decoding with cv2.imdecode. On
    # Windows OpenCV builds, cv2.imread can fail or emit warnings when the
    # path contains non-ASCII characters. Reading bytes with Python and
    # using cv2.imdecode avoids that issue.
    try:
        with open(path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        # If reading bytes fails for some reason, fall back to cv2.imread
        # (this may emit a warning on Windows with Unicode paths).
        pass

    # Final fallback: let OpenCV try to read the path directly.
    try:
        img = cv2.imread(path)
        return img
    except Exception:
        return None


class CornerFinder:
    def __init__(self, image_folder):
        # Open folder containing images
        self.image_folder = image_folder
        # Read document with photos stats in the folder. Accept either
        # 'photos_data.csv' (new) or fall back to 'photos.csv' (older name).
        csv_path_a = os.path.join(self.image_folder, 'photos_data.csv')
        csv_path_b = os.path.join(self.image_folder, 'photos.csv')
        if os.path.exists(csv_path_a):
            csv_path = csv_path_a
        elif os.path.exists(csv_path_b):
            csv_path = csv_path_b
        else:
            raise FileNotFoundError(f"Could not find photos CSV in '{self.image_folder}': looked for photos_data.csv and photos.csv")
        photos_df = pd.read_csv(csv_path)

        # Find marker type and size
        marker_size_dict = {"1p":8.2,"2e":5.3,"3e":4.35,"2v":4.2,"3v":4.2}
        self.marker_type = photos_df["marker_type"].iloc[0]
        print(f"[INFO] Marker type: {self.marker_type}")
        self.marker_size = marker_size_dict[self.marker_type]
        #print(f"[INFO] Marker size: {self.marker_size} mm")

        # Create output csv file
        self.output_file = f'corners_{self.marker_type}.csv'

        # ArUco setup
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # Create DetectorParameters with version compatibility
        try:
            # Newer OpenCV: DetectorParameters is a class
            self.arucoParams = cv2.aruco.DetectorParameters()
        except Exception:
            # Older OpenCV: use factory function
            self.arucoParams = cv2.aruco.DetectorParameters_create()

        # Try to set corner refinement method if available
        try:
            self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except Exception:
            # If attribute not present, ignore — it's optional
            pass

        # Prefer the ArucoDetector object when available (new API). If not,
        # leave detector as None and fall back to cv2.aruco.detectMarkers.
        if hasattr(cv2.aruco, 'ArucoDetector'):
            try:
                self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)
            except Exception:
                self.detector = None
        else:
            self.detector = None

    def transform_pf_to_cf(self, px,py,pz,pe):
        
        # PF0 in CF coordinates
        # PF0_inCF = np.array([153.865,-137.38,38.24+2.5]) #mm original
        PF0_inCF = np.array([153.865,-137.38,38.24+2.5+6.0]) #mm only z correction
        # PF0_inCF = np.array([153.865+0.44689565,-137.38+1.1555734,38.24+2.5+6.07420073]) #mm


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

        cfRcmf = cfRpf @ pfRi1 @ i1Ri0 @ i0Rcmf
        rvec_cf, _ = cv2.Rodrigues(cfRcmf)
        rvec_cf = rvec_cf.flatten()  # Convert from (3,1) to (3,) for cleaner CSV


        pz = -1 * pz  #X axis is inverted because relative to Cam F, right handed Print F requires this
        tvec_cf = (cfRpf @ np.array([[py],[px],[pz]])).flatten() + PF0_inCF

        # print('YO: ',pe)
        # pfRcmf = pfRi1 @ i1Ri0 @ i0Rcmf
        # print( cfRcmf @ np.array([[1],[1],[1]]))

        # print('TVEC: ',tvec_cf)
        return rvec_cf, tvec_cf
    
    def find_corners(self, image_path):
        # Load image and convert to grayscale
        # Try to load the image, using a fallback for problematic paths
        image = imread_with_fallback(image_path)

        # Guard against failed image loads (this causes the cvtColor assertion)
        if image is None:
            # Provide a clear, actionable error
            raise FileNotFoundError(
                f"Could not load image at '{image_path}'.\n"
                "Check that the path is correct and the file exists, and that OpenCV supports the image format.\n"
                "If the path contains non-ASCII characters, try moving files to a path with ASCII-only names or enable the imdecode fallback.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the image
        corners, ids, rejected = self.detector.detectMarkers(gray)
        # Determine if id is valid for the given marker type
        match self.marker_type:
            case "1p":
                expected_ids = [5]
            case "2e":
                expected_ids = [1, 0]
            case "3e":
                expected_ids = [2, 3, 4]
            case "2v":
                expected_ids = [6, 7]
            case "3v":
                expected_ids = [8, 9, 10]
            case _:
                raise ValueError(f"Unknown marker type: {self.marker_type}")
            
        # Only save corners for valid ids
        valid_corners = []
        valid_ids = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in expected_ids:
                    valid_corners.append(corners[i])
                    valid_ids.append(marker_id)

        return valid_corners, valid_ids

    def save_output(self, results, output_path=None):
        # Save data to CSV
        # Read document with photos stats (use the CSV from the image folder)
        csv_path = os.path.join(self.image_folder, 'photos_data.csv')
        photos_df = pd.read_csv(csv_path)

        # Process photos_df to extract relevant information
        # Finding pose for each image
        pose_split = photos_df["pose"].str.split(";", expand=True).astype(float)
        photos_df["x"] = pose_split[0]
        photos_df["y"] = pose_split[1]
        photos_df["z"] = pose_split[2]
        photos_df["e"] = pose_split[3]

        # Convert from printer frame to camera frame
        rvecs = []
        tvecs = []
        for index, row in photos_df.iterrows():
            rvec, tvec = self.transform_pf_to_cf(row["x"], row["y"], row["z"], row["e"])
            rvecs.append(rvec)
            tvecs.append(tvec)
        
        # Output file: img_id;marker_type;intensity;folder_name;pose;rvec;tvec;n_valid;corners;ids
        # `results` is expected to be a dict mapping img_id (string) -> (valid_corners, valid_ids)
        # Determine output file path. If output_path is provided, use it. Otherwise
        # write to the default `self.output_file` inside the image folder.
        if output_path is None:
            out_full = os.path.join(self.image_folder, self.output_file)
        else:
            out_full = output_path

        os.makedirs(os.path.dirname(out_full) or '.', exist_ok=True)

        with open(out_full, 'w') as f:
            f.write('img_id;marker_type;intensity;folder_name;pose;rvec;tvec;n_valid;corners;ids\n')
            folder_name = os.path.basename(self.image_folder)
            for i, row in photos_df.iterrows():
                raw_id = row["img_id"]
                # Helper to find matching key in results (handles leading-zero differences)
                def lookup_results(key_raw):
                    # direct string lookup
                    kstr = str(key_raw)
                    if kstr in results:
                        return results[kstr]

                    # try numeric matching against existing result keys
                    try:
                        kval = int(key_raw)
                    except Exception:
                        return ([], [])

                    for k in results.keys():
                        try:
                            if int(k) == kval:
                                return results[k]
                        except Exception:
                            continue

                    return ([], [])

                img_id = str(raw_id)
                marker_type = row.get("marker_type", '')
                intensity = row.get("intensity", '')
                pose = row.get("pose", '')
                #change ; to , inside pose
                pose = pose.replace(';', ',')
                rvec = ','.join(map(str, rvecs[i]))
                tvec = ','.join(map(str, tvecs[i]))

                # Lookup per-image detection results; default to empty lists
                valid_corners, valid_ids = lookup_results(raw_id)

                # number of valid detected markers (0,1,...)
                n_valid = len(valid_ids)

                # Concatenate corners for all valid markers into a single field.
                # Each marker's 4 corner points are flattened and joined by commas;
                # different markers are separated by the pipe character `|`.
                if n_valid == 0:
                    corners_str = ''
                    ids_str = ''
                else:
                    corners_str = '|'.join([','.join(map(str, corner.flatten())) for corner in valid_corners])
                    ids_str = '|'.join(map(str, valid_ids))

                # Write line with semicolon separators so embedded commas in rvec/tvec/corners
                # don't break the CSV structure.
                f.write(f'{img_id};{marker_type};{intensity};{folder_name};{pose};{rvec};{tvec};{n_valid};{corners_str};{ids_str}\n')

        
# Main function to run the corner finder
if __name__ == "__main__":
    # For each folder inside this folder
    main_folder = "C:\\Users\\eduar\\OneDrive\\Área de Trabalho\\bepe\\codes\\markers\\data\\d100"  # Replace with your image folder path
    # Get all subfolders, but skip the 'results' folder to avoid re-processing outputs
    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir() and os.path.basename(f.path) != 'results']
    for image_folder in subfolders:
        print(f"Processing folder: {image_folder}")
        corner_finder = CornerFinder(image_folder)

        # Loop through images in the folder and collect per-image detection results
        results = {}
        for image_name in os.listdir(image_folder):
            if image_name.lower().endswith('.jpg'):
                print(image_name)
                image_path = os.path.join(image_folder, image_name)
                valid_corners, valid_ids = corner_finder.find_corners(image_path)
                img_id = os.path.splitext(image_name)[0]
                results[img_id] = (valid_corners, valid_ids)

        # Ensure results directory exists and save output there (one file per subfolder)
        results_dir = os.path.join(main_folder, 'results')
        os.makedirs(results_dir, exist_ok=True)
        out_filename = f'corners_{corner_finder.marker_type}_{os.path.basename(image_folder)}.csv'
        out_path = os.path.join(results_dir, out_filename)
        corner_finder.save_output(results, out_path)