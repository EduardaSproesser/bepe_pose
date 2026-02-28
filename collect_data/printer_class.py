import serial
from serial.tools import list_ports
import time
import sys
import math
import re
from timeit import default_timer as tm

PRINTER_MIN_SLEEP = 0.1

def _remove_comment(string):
    if string.find(';')==-1:
        return string
    else:
        return string[:string.index(';')]

class Printer:
    def __init__(self, valid_pos_check_func, wrench_overload_check_func, portname=None):
        if portname is None:
            ports = list(list_ports.comports())
            for p in ports:
                if ":7523" in p[2]:
                    print(f"Printer found on {p[0]}.")
                    portname = p[0]
            if portname is None:
                print("Printer not found. Please Check connections.")
                sys.exit()
        try:
            self.port = serial.Serial(portname, 115200)
            print("Driver initialized. Port opened.")
            time.sleep(1.5)
        except serial.SerialException:
            print("Could not open port.")
            sys.exit(0)
        self.valid_pos_check_func = valid_pos_check_func
        self.wrench_overload_check_func = wrench_overload_check_func

    def send_line(self, line, read_output = False):
        l = line.strip()
        l = _remove_comment(l)
        if l.isspace()==False and len(l)>0:
            self.port.reset_input_buffer()
            self.port.write((l + "\r\n").encode())
            if read_output:
                sensor_out = self.port.readline()
                return sensor_out
            else:
                return 1

    def send_file(self, file, read_output=False):
        self.port.write(b'\r\n\r\n')
        time.sleep(2)
        self.port.reset_input_buffer()
        for line in file.splitlines():
            output = self.send_line(line, read_output=read_output)
            if output and read_output:
                print(output.decode('utf-8').strip())

    def send_blocking(self, gcode, timeout=20):
        if re.search("M118 .+", gcode) is not None:
            print("Error: gcode already contains M118 instructions.")
            return 1
        self.port.timeout = timeout
        self.port.reset_input_buffer()
        self.port.reset_output_buffer()
        self.send_file(gcode + "M118 done\n")
        t0 = tm()
        while tm() - t0 <= timeout:
            line = self.port.readline().decode('UTF-8')
            if re.search("done", line):
                 return "done"
        print(f"Error: sendBlocking reached timeout of {timeout}s.")
        return 1

    def send_go_to(self, x,y,z, e, f=4000, timeout=10):
        time.sleep(PRINTER_MIN_SLEEP)
        if self.valid_pos_check_func is not None and not self.valid_pos_check_func(x,y,z,e):
            print(x,y,z,e)
            print("Error: position is outside allowed range for the specified d_max.\n", flush=True)
            return 1
        gcode = (f"G90\n"
                 f"G1 F{f:f} X{x:f} Y{y:f} Z{z:f} E{e:f}\n"
                 f"M400")
        self.port.timeout = timeout
        self.port.reset_input_buffer()
        self.port.reset_output_buffer()
        self.send_line(gcode)
        t0 = tm()
        while tm() - t0 <= timeout:
            line = self.port.readline().decode('UTF-8')
            if line == "ok\n": return
        print(f"Error: sendGoTo reached timeout of {timeout}s.")
        return 1

    def close(self):
        self.port.close()