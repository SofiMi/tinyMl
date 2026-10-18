#!/usr/bin/env python3
"""Select the console input/output interface in an existing ESP-IDF config."""
import argparse
from datetime import datetime
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
UART0_SETTINGS = {
    'CONFIG_ESP_CONSOLE_UART_DEFAULT': 'y',
    'CONFIG_ESP_CONSOLE_USB_CDC': None,
    'CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG': None,
    'CONFIG_ESP_CONSOLE_UART_CUSTOM': None,
    'CONFIG_ESP_CONSOLE_NONE': None,
    'CONFIG_ESP_CONSOLE_UART_BAUDRATE': '115200',
    'CONFIG_ESP_CONSOLE_SECONDARY_NONE': None,
    'CONFIG_ESP_CONSOLE_SECONDARY_USB_SERIAL_JTAG': 'y',
}

USB_SETTINGS = dict(UART0_SETTINGS, **{
    'CONFIG_VFS_SUPPORT_IO': 'y',
    'CONFIG_ESP_CONSOLE_UART_DEFAULT': None,
    'CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG': 'y',
    'CONFIG_ESP_CONSOLE_SECONDARY_NONE': 'y',
    'CONFIG_ESP_CONSOLE_SECONDARY_USB_SERIAL_JTAG': None,
})


def configure(path, interface):
    if interface not in ('usb', 'uart0'):
        raise ValueError(f'Unsupported console interface: {interface}')
    settings = USB_SETTINGS if interface == 'usb' else UART0_SETTINGS
    existed = path.exists()
    source = path.read_text() if existed else (ROOT / 'sdkconfig.defaults').read_text()
    kept = []
    for line in source.splitlines():
        match = re.match(r'(?:# )?(CONFIG_\w+)(?:=| is not set)', line)
        if not match or match[1] not in settings:
            kept.append(line)
    for key, value in settings.items():
        kept.append(f'{key}={value}' if value is not None else f'# {key} is not set')
    result = '\n'.join(kept) + '\n'
    if existed and result == source:
        print(f'Already configured: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if existed:
        backup = path.with_name(path.name + f'.before-{interface}-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        # Exclusive creation preserves every previous backup.
        with backup.open('x') as output:
            output.write(source)
        print(f'Backup: {backup}')
    path.write_text(result)
    description = ('native USB Serial/JTAG input/output' if interface == 'usb' else
                   'UART0 115200 input/output; USB Serial/JTAG output mirror')
    print(f'Configured {path}: {description}')
    print('Rebuild and flash using the same SDKCONFIG and build directory.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', type=Path, default=ROOT / 'sdkconfig')
    parser.add_argument('--interface', choices=('usb', 'uart0'), required=True,
                        help='usb for Espressif 303a:1001; uart0 for a USB-UART bridge')
    args = parser.parse_args()
    configure(args.config, args.interface)
