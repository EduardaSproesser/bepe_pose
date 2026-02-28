import subprocess
import os
import sys
from pathlib import Path

COMMANDS = {
    "d0": "5505ff0700326d",
    "d50": "5505ff07022677",
    "d100": "5505ff0703e8b4",
    "red": "5503ff09000003e8b4",
    "green": "5503ff09007803e83c",
    "blue": "5503ff0900f003e8c4",
    "white": "5503ff09000000009f"
}

class LedBLE:
    def __init__(self):
        pass

    def test_connection(self):
        # Só printa, não conecta de fato
        print("Test connection skipped on Windows. Commands will run via subprocess.")

    def set_color_intensity(self, value):
        """Recebe key do COMMANDS ou HEX string e chama led_send.py"""
        if value in COMMANDS:
            hex_code = COMMANDS[value]
        else:
            hex_code = value
        print(f"Sending command: {hex_code}")
        # Chama subprocess para enviar BLE
        repo_root = Path(os.environ.get("BEPE_ROOT", Path(__file__).resolve().parents[1]))
        led_send_script = Path(
            os.environ.get(
                "BEPE_LED_SEND_SCRIPT",
                str(repo_root / "collect_data" / "led_send.py")
            )
        )
        subprocess.run([sys.executable, str(led_send_script), hex_code], check=False)
