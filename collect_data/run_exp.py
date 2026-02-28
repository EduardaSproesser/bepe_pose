import csv
import os
from printer_class import Printer
from camera_marker_capture import CameraMarkerCapture
from led_ble import LedBLE
import time
import math
from pathlib import Path

# === Extruder step configuration ===
DEGREES_PER_STEP = 10
X_MAX = 300
Y_MAX = 300
Z_MAX = 160


# ---------------- UTILS ----------------
def degrees_to_mm(degrees: float) -> float:
    """
    Convert motor shaft rotation (degrees) to linear Z displacement (mm).
    """
    return (degrees / 360.0) * (20 * math.pi)



def check_valid_pos(x, y, z, e):
    return all([
        0 <= x <= X_MAX,
        0 <= y <= Y_MAX,
        0 <= z <= Z_MAX,
    ])



def snail_order(x_list, y_list):
    points = []
    for i, y in enumerate(y_list):
        if i % 2 == 0:
            for x in x_list:
                points.append((x, y))
        else:
            for x in reversed(x_list):
                points.append((x, y))
    return points


# ---------------- PRINTER OPS ----------------

def calibrate_extruder(printer):
    printer.send_blocking("M302 P1\n")
    time.sleep(1)
    printer.send_blocking(f"M92 E{(16 * 200 / (20 * math.pi)):f}\n")
    print("=== Extruder Calibration ===")
    print("1. Use the printer knob to align the extruder motor.")
    input("2. Press Enter when aligned... ")
    printer.send_blocking("G92 E0\n")
    print("Extruder home (E=0) set.")



def autohome(printer):
    print("Sending printer to autohome (G28)...")
    printer.send_blocking("G28\n")
    # In order to improve positional accuracy, the homing procedure can re-bump at a slower speed according to the [XYZ]_HOME_BUMP_MM and HOMING_BUMP_DIVISOR settings.
    # This is especially useful for the Z axis, which may have more mechanical play.
    # Y_HOME_BUMP_MM and HOMING_BUMP_DIVISOR can be adjusted for finer control.
    # printer.send_blocking("M203 Y2\n")  # Set Y max feedrate (mm/s) to a lower value (e.g., 2 mm/s)
    # printer.send_blocking("M201 Y20\n") # Set Y max acceleration (mm/s^2) to a lower value (e.g., 20 mm/s^2)
    # printer.send_blocking("G28\n")
    # # Change speed back to normal
    # printer.send_blocking("M203 Y10\n")  # Reset Y max feedrate (mm/s) to default value (e.g., 10 mm/s)
    # printer.send_blocking("M201 Y100\n") # Reset Y max acceleration (mm/s^2) to default value (e.g., 100 mm/s^2)
    time.sleep(2)
    printer.send_blocking("G92 X0 Y0 Z0 E0\n")

# ---------------- EXPERIMENT ROUTINE ----------------

def save_photo_csv(save_folder, row, experiment_name):
    csv_path = os.path.join(save_folder, f"photos_{experiment_name}.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["img_id", "marker_type", "color", "intensity", "camera_index", "run_number", "pose", "timestamp"])
        writer.writerow(row)



def run_experiment(printer, camera, led, exp, number_run):
    exp_id = exp["ID"]
    print(f"\n=== Running Experiment {exp['Experiment']} (ID={exp_id}) ===")
    marker_type = "1p"
    color = "white"
    intensity = "d100"
    led.set_color_intensity(color)
    led.set_color_intensity(intensity)
    calibrate_extruder(printer)
    autohome(printer)
    time.sleep(30)
    X_START = 110
    Y_START = 10
    x_vals = list(range(X_START, X_MAX + 1, 50))
    y_vals = list(range(Y_START, Y_MAX + 1, 50))
    z_vals = list(range(0, Z_MAX + 1, 50))
    e_angles = list(range(130, 230, DEGREES_PER_STEP))
    prev_z = None
    prev_x = None
    prev_y = None
    for z in z_vals:
        print(f"Z position: {z}")
        xy_points = snail_order(x_vals, y_vals)
        for x, y in xy_points:
            if y > 195:
                e_end = 185
                e_angles_mod = [deg for deg in e_angles if deg <= e_end]
            elif y < 95:
                e_start = 175
                e_angles_mod = [deg for deg in e_angles if deg >= e_start]
            else:
                e_angles_mod = e_angles
            for e_deg in e_angles_mod:
                e = degrees_to_mm(e_deg)
                print(f"Moving to X={x}, Y={y}, Z={z}, E={e_deg}° ({e:.4f} mm)")
                
                printer.send_go_to(x, y, z, e)
                
                # Ajust delay based on movement
                if prev_z != z:
                    delay = 12
                elif prev_x != x or prev_y != y:
                    delay = 3
                else:
                    delay = 1
                    
                # If first move, wait longer
                if prev_z is None and prev_x is None and prev_y is None:
                    delay = 25

                time.sleep(delay)
                
                filepath, photo_row = camera.capture_and_save_photo(
                    marker_type, color, intensity, 1, [x, y, z, e_deg], number_run
                )
                if filepath and photo_row:
                    save_photo_csv(camera.save_folder, photo_row, exp["Experiment"])
                prev_z = z
                prev_x = x
                prev_y = y
    print(f"Experiment {exp_id} complete.")
    exp["Status"] = "DONE"


# ---------------- MAIN ----------------

def load_experiments(txt_file):
    experiments = []
    with open(txt_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                experiments.append({"Experiment": parts[0], "ID": parts[1], "Status": parts[2]})
    return experiments



def save_experiments(txt_file, experiments):
    with open(txt_file, "w") as f:
        for exp in experiments:
            f.write(f"{exp['Experiment']},{exp['ID']},{exp['Status']}\n")



def main():
    repo_root = Path(os.environ.get("BEPE_ROOT", Path(__file__).resolve().parents[1]))
    experiments_file = Path(
        os.environ.get("BEPE_EXPERIMENTS_FILE", str(repo_root / "collect_data" / "experiments.txt"))
    )

    printer = Printer(valid_pos_check_func=check_valid_pos, wrench_overload_check_func=None)
    camera = CameraMarkerCapture()
    led = LedBLE()
    camera.set_experiment_folder("experiments")
    camera.init_camera(1)
    camera.cap.set(3, 1280)
    camera.cap.set(4, 960)
    experiments = load_experiments(experiments_file)
    for exp in experiments:
        if exp["Status"].upper() == "DONE":
            continue
        print(f"\nNext experiment: {exp['Experiment']} (ID={exp['ID']}) - Status={exp['Status']}")
        start = input("Start this experiment? (y/n): ")
        if start.lower() != "y":
            continue
        number_run = sum(1 for e in experiments if e["Experiment"] == exp["Experiment"] and e["Status"] == "DONE") + 1
        run_experiment(printer, camera, led, exp, number_run)
        save_experiments(experiments_file, experiments)
    printer.close()
    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
