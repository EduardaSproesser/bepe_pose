# led_send.py
import sys
import asyncio
from bleak import BleakClient

# Recebe o comando HEX como argumento
if len(sys.argv) < 2:
    print("Usage: python led_send.py <hex_command>")
    sys.exit(1)

hex_command = sys.argv[1]

# BLE config
ADDRESS = "41:42:39:3C:81:4B"
CHAR_UUID = "0000fff3-0000-1000-8000-00805f9b34fb"

async def main():
    try:
        async with BleakClient(ADDRESS) as client:
            if client.is_connected:
                print(f"Connected to {ADDRESS}, sending command...")
                await client.write_gatt_char(CHAR_UUID, bytes.fromhex(hex_command), response=False)
                print("Command sent!")
            else:
                print("Failed to connect!")
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
