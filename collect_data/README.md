# Data Collection (`collect_data`)

This folder contains scripts used to run automated data-collection experiments with:

- a 3D printer (motion control via G-code over serial),
- a BLE LED controller (lighting condition control), and
- a camera capture pipeline (photo + metadata logging).

## Files

### `run_exp.py`

Main orchestration script for full experiments.

What it does:

- Loads experiment definitions from a text file.
- Initializes printer, camera, and LED controller.
- Runs extruder calibration and printer homing.
- Moves through a grid of positions (`X`, `Y`, `Z`) plus extruder-angle-derived `E` positions.
- Captures photos and saves metadata rows to CSV.
- Marks finished experiments as `DONE`.

Important details from current implementation:

- Uses a “snail” scan order for XY points.
- Uses dynamic wait time based on whether movement changed `Z`, `X/Y`, or only `E`.
- Stores photo metadata as:
  - `img_id`, `marker_type`, `color`, `intensity`, `camera_index`, `run_number`, `pose`, `timestamp`

---

### `printer_class.py`

Serial wrapper for printer communication.

Capabilities:

- Auto-detects printer port (looks for USB descriptor containing `:7523`).
- Sends single G-code lines and multi-line blocks.
- Supports blocking commands with timeout (`send_blocking`).
- Sends absolute `X/Y/Z/E` moves with boundary checks (`send_go_to`).

Configured limits in `run_exp.py`:

- `X_MAX = 300`
- `Y_MAX = 300`
- `Z_MAX = 160`

---

### `led_ble.py`

High-level LED helper with predefined BLE command presets:

- intensity presets: `d0`, `d50`, `d100`
- color presets: `red`, `green`, `blue`, `white`

`set_color_intensity()` accepts either preset keys or a raw HEX command.

---

### `led_send.py`

Low-level BLE sender using `bleak`.

- Connects to a fixed BLE MAC address.
- Writes a HEX command to a fixed GATT characteristic.

Current constants:

- `ADDRESS = "41:42:39:3C:81:4B"`
- `CHAR_UUID = "0000fff3-0000-1000-8000-00805f9b34fb"`

---

### `experiments.txt`

Experiment queue file.

Expected format per line:

```text
ExperimentName,ID,Status
```

Example:

```text
5,3,
6,3,DONE
```

## Required Setup Before Running

Paths are now root-configurable through environment variables.

1. Optional: set repository root (default is auto-detected from script location):

	```bash
	set BEPE_ROOT=C:\path\to\bepe_pose
	```

2. Optional: set a custom experiments file path:

	```bash
	set BEPE_EXPERIMENTS_FILE=C:\path\to\experiments.txt
	```

3. Optional: set a custom LED sender script path:

	```bash
	set BEPE_LED_SEND_SCRIPT=C:\path\to\led_send.py
	```

4. Ensure `camera_marker_capture.py` is available in your Python path.

	- `run_exp.py` imports `CameraMarkerCapture`, but this file is not present in this repository folder structure.

## Dependencies

Install the Python packages used by these scripts:

```bash
pip install pyserial bleak
```

Also ensure your camera module dependencies for `CameraMarkerCapture` are installed (likely `opencv-python` / `numpy`, depending on its implementation).

## How to Run

From this folder:

```bash
python run_exp.py
```

Interactive flow:

- Script shows next pending experiment.
- Confirm start with `y`.
- System executes motion + capture routine.
- Status is updated and saved.

## Practical Notes

- This workflow controls real hardware. Keep emergency stop access available.
- Validate movement limits and coordinate frames before long runs.
- Start with a reduced grid and fewer `E` angles as a smoke test.
