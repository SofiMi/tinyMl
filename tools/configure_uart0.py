#!/usr/bin/env python3
"""Legacy UART0 helper for USB-UART bridges; use configure_console.py for native USB."""
import argparse
from pathlib import Path
from configure_console import ROOT, UART0_SETTINGS as SETTINGS, configure as configure_console


def configure(path):
    configure_console(path, 'uart0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', type=Path, default=ROOT / 'sdkconfig')
    configure(parser.parse_args().config)
