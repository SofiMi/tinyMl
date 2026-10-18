#!/usr/bin/env python3
"""Regression: the observed 303a:1001 port needs native USB as primary input."""
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def enabled(text, key):
    return f'{key}=y' in text.splitlines()


class ConsoleConfigTests(unittest.TestCase):
    def test_defaults_accept_input_on_observed_native_usb(self):
        text = (ROOT / 'sdkconfig.defaults').read_text()
        self.assertTrue(enabled(text, 'CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG'),
                        '303a:1001 needs PRIMARY USB; the secondary mirror cannot receive commands')
        self.assertTrue(enabled(text, 'CONFIG_VFS_SUPPORT_IO'))
        self.assertFalse(enabled(text, 'CONFIG_ESP_CONSOLE_UART_DEFAULT'))
        self.assertTrue(enabled(text, 'CONFIG_ESP_CONSOLE_SECONDARY_NONE'))

    def test_migrate_uart_mirror_to_native_and_back(self):
        from configure_console import configure
        original = ('CONFIG_ESP_CONSOLE_UART_DEFAULT=y\n'
                    '# CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG is not set\n'
                    'CONFIG_ESP_CONSOLE_SECONDARY_USB_SERIAL_JTAG=y\n'
                    '# CONFIG_VFS_SUPPORT_IO is not set\n'
                    'CONFIG_TINYML_OLED_SDA=13\nCONFIG_CUSTOM_VALUE=42\n')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sdkconfig'
            path.write_text(original)
            configure(path, 'usb')
            text = path.read_text()
            self.assertTrue(enabled(text, 'CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG'))
            self.assertTrue(enabled(text, 'CONFIG_VFS_SUPPORT_IO'))
            self.assertFalse(enabled(text, 'CONFIG_ESP_CONSOLE_UART_DEFAULT'))
            self.assertFalse(enabled(text, 'CONFIG_ESP_CONSOLE_SECONDARY_USB_SERIAL_JTAG'))
            self.assertTrue(enabled(text, 'CONFIG_ESP_CONSOLE_SECONDARY_NONE'))
            self.assertIn('CONFIG_TINYML_OLED_SDA=13\nCONFIG_CUSTOM_VALUE=42\n', text)
            backups = list(Path(folder).glob('*.before-usb-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), original)
            configure(path, 'usb')
            self.assertEqual(path.read_text(), text)
            self.assertEqual(list(Path(folder).glob('*.before-usb-*')), backups)
            configure(path, 'uart0')
            self.assertTrue(enabled(path.read_text(), 'CONFIG_ESP_CONSOLE_UART_DEFAULT'))
            self.assertTrue(enabled(path.read_text(), 'CONFIG_ESP_CONSOLE_SECONDARY_USB_SERIAL_JTAG'))
            self.assertFalse(enabled(path.read_text(), 'CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG'))


if __name__ == '__main__':
    unittest.main()
