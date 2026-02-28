# Camera Calibration Utilities (`calib_cam`)

This folder contains scripts to:

1. capture calibration images,
2. calibrate a camera (including omnidirectional/fisheye-like models via OpenCV omnidir), and
3. generate printable ArUco markers in PDF format.

## Files Overview

### `take_calib_photos.py`

Interactive image capture utility for Windows (`cv2.CAP_DSHOW`) with automatic camera detection and resolution probing.

Main features:

- Scans camera indices (`0..9`) and tests multiple resolutions.
- Lets you confirm the correct camera in a preview window.
- Captures JPEG images at high quality (`IMWRITE_JPEG_QUALITY = 95`).

Capture modes:

1. **Manual capture**
	- Key `SPACE`: save current frame.
	- Key `q`: quit.
	- Output folder: `photos/`

2. **Sequence with preview**
	- Asks for number of photos and interval.
	- Shows countdown/preview between captures.
	- Includes reconnection logic if camera fails.
	- Output folder: `photos_sequence/`

3. **Fast/stable sequence (no preview)**
	- Opens and closes the camera on each shot to improve robustness.
	- Useful when preview mode is unstable.
	- Output folder: `photos_fast_sequence/`

Run:

```bash
python take_calib_photos.py
```

---

### `workspace_calibration_final.py`

Camera calibration script using checkerboard images.

Current behavior in the script:

- Reads images from: `calib_new_photos/*.jpg`
- Checkerboard size: `CHECKERBOARD = (10, 15)` (inner corners)
- Detects corners with `cv2.findChessboardCorners`
- Refines corners with `cv2.cornerSubPix`
- Uses **OpenCV omnidir calibration** (`cv2.omnidir.calibrate`) when `omnidir=True`
- Prints:
  - per-image reprojection error,
  - global MRMS/RMS,
  - estimated camera parameters (`K`, `XI`, `D`),
  - and basic success/failure summary

Optional branch:

- `rectilinear=True` enables standard `cv2.calibrateCamera` path.

Run:

```bash
python workspace_calibration_final.py
```

Notes:

- All calibration images must have the same dimensions.
- `cv2.omnidir` requires OpenCV Contrib modules.
- You will likely need to adapt image folder and checkerboard pattern to your dataset.

---

### `create_pdf.py`

Generates an A4 PDF containing ArUco markers (`DICT_4X4_50`) with configurable final printed size.

Current configuration in file:

- Marker groups:

  ```python
  groups = [
		(1, 12),
		(1, 10),
  ]
  ```

  meaning: generate 1 marker of 12 cm and 1 marker of 10 cm.

- Border width is set to `final_size_cm / 8`.
- IDs are unique and sequential (`0..49` max for `DICT_4X4_50`).
- Draws dashed cut guides around each marker.
- Output PDF: `aruco_markers_fapesp.pdf`

Run:

```bash
python create_pdf.py
```

## Environment / Dependencies

Recommended packages:

```bash
pip install opencv-contrib-python numpy matplotlib reportlab
```

Why `opencv-contrib-python`:

- `cv2.omnidir` and `cv2.aruco` are used in this folder.

## Typical Workflow

1. Print ArUco markers if needed (`create_pdf.py`).
2. Capture checkerboard/calibration images (`take_calib_photos.py`).
3. Put selected images in the calibration input folder (currently `calib_new_photos/`).
4. Run calibration (`workspace_calibration_final.py`).
5. Copy resulting camera parameters (`K`, `XI`, `D`) to the next processing stage.
