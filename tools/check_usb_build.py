#!/usr/bin/env python3
"""Check a built ESP-IDF native USB console, including the actual linked I/O driver."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def check(build):
    for name in ('config/sdkconfig.h', 'bootloader/config/sdkconfig.h'):
        config = (build / name).read_text()
        assert '#define CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG 1' in config, name
        assert '#define CONFIG_ESP_CONSOLE_UART_DEFAULT 1' not in config, name
        assert '#define CONFIG_ESP_CONSOLE_SECONDARY_NONE 1' in config, name
    config = (build / 'config/sdkconfig.h').read_text()
    assert '#define CONFIG_VFS_SUPPORT_IO 1' in config, 'VFS I/O must be enabled'
    project = json.loads((build / 'project_description.json').read_text())
    for component in ('vfs', 'esp_stdio', 'esp_driver_usb_serial_jtag'):
        assert component in project['build_components'], component
    cache = (build / 'CMakeCache.txt').read_text()
    match = re.search(r'^CMAKE_NM:FILEPATH=(.+)$', cache, re.M)
    assert match, 'CMAKE_NM missing from build cache'
    result = subprocess.run([match[1], '--defined-only', str(build / project['app_elf'])],
                            text=True, capture_output=True, check=True)
    symbols = {line.split()[-1] for line in result.stdout.splitlines() if line.split()}
    for symbol in ('esp_stdio_install_io_driver', 'usb_serial_jtag_driver_install',
                   'usb_serial_jtag_vfs_dev_port_init', 'usbjtag_rx_char_via_driver',
                   'usbjtag_tx_char_via_driver', 'esp_stdio_register'):
        assert symbol in symbols, f'Missing real console driver: {symbol}'
    print('PASS: primary native USB in app/bootloader, VFS enabled, real USB driver linked')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', nargs='?', type=Path, default=Path('build/esp32ai'))
    check(parser.parse_args().build)
