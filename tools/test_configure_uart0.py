#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
from configure_uart0 import configure, SETTINGS


class ConfigureUART0Tests(unittest.TestCase):
    def test_migration_preserves_other_settings_and_original(self):
        original = ('CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y\n'
                    '# CONFIG_ESP_CONSOLE_UART_DEFAULT is not set\n'
                    'CONFIG_ESP_CONSOLE_SECONDARY_NONE=y\n'
                    'CONFIG_ESP_CONSOLE_UART_BAUDRATE=9600\n'
                    'CONFIG_TINYML_OLED_SDA=13\nCONFIG_CUSTOM_VALUE=42\n')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sdkconfig'
            path.write_text(original)
            configure(path)
            result = path.read_text()
            self.assertIn('CONFIG_ESP_CONSOLE_UART_DEFAULT=y\n', result)
            self.assertIn('CONFIG_ESP_CONSOLE_UART_BAUDRATE=115200\n', result)
            self.assertIn('# CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG is not set\n', result)
            self.assertIn('CONFIG_ESP_CONSOLE_SECONDARY_USB_SERIAL_JTAG=y\n', result)
            self.assertIn('CONFIG_TINYML_OLED_SDA=13\nCONFIG_CUSTOM_VALUE=42\n', result)
            backups = list(Path(folder).glob('*.before-uart0-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), original)
            configure(path)
            self.assertEqual(path.read_text(), result)
            self.assertEqual(list(Path(folder).glob('*.before-uart0-*')), backups)

    def test_new_config_keeps_board_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'build' / 'config'
            configure(path)
            result = path.read_text()
            self.assertIn('CONFIG_SPIRAM_MODE_OCT=y\n', result)
            self.assertIn('CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y\n', result)
            for key, value in SETTINGS.items():
                expected = f'{key}={value}' if value else f'# {key} is not set'
                self.assertEqual(result.splitlines().count(expected), 1)


if __name__ == '__main__':
    unittest.main()
