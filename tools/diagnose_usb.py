#!/usr/bin/env python3
"""Linux USB serial probe: collect identity/kernel logs and reproduce port-open errors.

Run in the activated ESP-IDF Python environment with all serial monitors closed.
Opening a serial port can toggle DTR/RTS and reset the board. No flash, driver
reload, USB reset, service stop, or package install is performed.
"""
import argparse
from datetime import datetime
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def command(args):
    print('\n$ ' + ' '.join(args), flush=True)
    if not shutil.which(args[0]):
        print('Not installed; skipped.')
        return None
    try:
        result = subprocess.run(args, text=True, errors='replace', capture_output=True,
                                timeout=10, cwd=ROOT)
        print(result.stdout, end='')
        print(result.stderr, end='')
        print(f'[exit {result.returncode}]')
        return result
    except subprocess.TimeoutExpired:
        print('Timed out after 10 seconds.')
        return None


def diagnose(port):
    print('Platform:', platform.platform())
    print('Python:', sys.executable)
    command(['git', 'rev-parse', '--short', 'HEAD'])
    command(['lsusb'])
    identity = command(['udevadm', 'info', '--query=path', '--name', port])
    if identity and identity.returncode == 0:
        device = Path('/sys' + identity.stdout.strip())
        for parent in (device, *device.parents):
            if (parent / 'idVendor').is_file():
                for name in ('idVendor', 'idProduct', 'manufacturer', 'product', 'speed'):
                    file = parent / name
                    if file.is_file():
                        print(f'USB {name}={file.read_text().strip()}')
                break
        driver = device / 'device/driver'
        if driver.exists():
            print('Driver:', driver.resolve())
    busy = command(['fuser', '-v', port])
    status = 1
    if busy and busy.returncode == 0:
        print('BUSY: close the listed monitor/application and rerun; nothing was killed.')
    else:
        try:
            import serial
        except ImportError:
            print('Missing pyserial: activate ESP-IDF and rerun with its python interpreter.')
            status = 2
        else:
            print(f'\nOpening {port} at 115200 (may reset board via DTR/RTS)...', flush=True)
            try:
                with serial.Serial(port, baudrate=115200, timeout=1,
                                   write_timeout=1, exclusive=True) as connection:
                    print('PASS: port opened and serial settings applied.')
                    print('First bytes:', repr(connection.read(256)))
                status = 0
            except (OSError, serial.SerialException):
                print('FAIL: serial open/configure/read failed; traceback identifies the failing operation.')
                traceback.print_exc(file=sys.stdout)
    return status


def kernel_log():
    print('\n$ dmesg --ctime (last 80 lines)', flush=True)
    try:
        result = subprocess.run(['dmesg', '--ctime'], text=True, errors='replace',
                                capture_output=True, timeout=10)
        if result.returncode == 0:
            print('\n'.join(result.stdout.splitlines()[-80:]))
        else:
            print(result.stderr.strip())
            print('Please also run: sudo dmesg --ctime | tail -80')
    except (OSError, subprocess.TimeoutExpired) as error:
        print(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('port', nargs='?', default='/dev/ttyACM0')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.exit(2, 'Run this on the Linux computer physically connected to the ESP32.\n')
    logs = ROOT / 'build/usb-diagnostics'
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.txt')
    original = sys.stdout
    try:
        with path.open('x') as log:
            sys.stdout = Tee(original, log)
            status = diagnose(args.port)
            kernel_log()
            print('\nReport:', path)
    finally:
        sys.stdout = original
    return status


if __name__ == '__main__':
    raise SystemExit(main())
