#!/usr/bin/env python3
"""Print flash usage from an IDF build, optionally request live /memory over USB."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[1]


def read_partitions(path):
    data = path.read_bytes()
    partitions = []
    for pos in range(0, len(data) - 31, 32):
        magic, kind, subtype, offset, size, label, flags = struct.unpack_from("<HBBII16sI", data, pos)
        if magic in (0xEBEB, 0xFFFF):  # MD5 record / erased padding
            break
        if magic != 0x50AA:
            raise ValueError(f"Invalid partition entry at {pos:#x}")
        partitions.append(dict(name=label.split(b"\0")[0].decode(), offset=offset,
                               capacity=size, type=kind, subtype=subtype))
    if not partitions:
        raise ValueError("Empty partition table")
    return partitions


def build_report(build):
    args = json.loads((build / "flasher_args.json").read_text())
    size_text = args["flash_settings"]["flash_size"]
    if not size_text.endswith("MB"):
        raise ValueError(f"Explicit flash size required, got {size_text}")
    capacity = int(size_text[:-2]) * 1024 ** 2
    table = args["partition-table"]
    parts = read_partitions(build / table["file"])
    images = {int(offset, 0): (build / name).resolve()
              for offset, name in args["flash_files"].items()}
    # Bootloader/header/table area is reserved even where no bytes are flashed.
    reserved_end = int(table["offset"], 0) + 0x1000
    end = reserved_end
    gaps = []
    for part in sorted(parts, key=lambda p: p["offset"]):
        start, length = part["offset"], part["capacity"]
        if start < end or start + length > capacity:
            raise ValueError("Overlapping partitions or partition exceeds configured flash")
        if start > end:
            gaps.append(dict(offset=end, bytes=start - end))
        end = start + length
        if start in images:
            used = images[start].stat().st_size
            if used > length:
                raise ValueError(f"Image exceeds partition {part['name']}")
            part.update(image_bytes=used, image_padding=length - used)
    if end < capacity:
        gaps.append(dict(offset=end, bytes=capacity - end))
    return dict(build=str(build.resolve()), flash_configured=capacity,
                reserved_boot_and_table=reserved_end, partitions=parts,
                unpartitioned=gaps)


def print_report(report):
    print(f"Build: {report['build']}")
    print(f"Configured flash: {report['flash_configured']:,} bytes (not a physical chip measurement)")
    print("Partition          Offset      Capacity       Image     Padding  (bytes)")
    for part in report["partitions"]:
        used = f"{part['image_bytes']:,}" if "image_bytes" in part else "unknown"
        padding = f"{part['image_padding']:,}" if "image_padding" in part else "unknown"
        print(f"{part['name']:<16} 0x{part['offset']:08x} {part['capacity']:11,} {used:>11} {padding:>11}")
    print(f"Reserved boot/header/table area: {report['reserved_boot_and_table']:,} bytes")
    print(f"Outside partitions: {sum(p['bytes'] for p in report['unpartitioned']):,} bytes")
    print("Padding is unused by the current image, not automatically usable RAM or a writable filesystem.")
    print("NVS/PHY contents and live free RAM cannot be inferred from build files.")
    print("For static firmware SRAM: idf.py -B <build-directory> size")
    print("For live heaps after model load: /memory in the device console, or --port <port> here.")
    print("\n300,000,000 parameters, weights only (no scales, metadata, KV or scratch):")
    for bits in (16, 8, 4, 2, 1):
        size = 300_000_000 * bits // 8
        print(f"{bits:2}-bit: {size:,} bytes = {size / 1024**2:.2f} MiB")


def capture(port, timeout, output):
    try:
        import serial
    except ImportError:
        raise SystemExit("For --port install pyserial in this Python environment")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Do not deliberately reset the board or enter its bootloader. The OS may
    # still toggle lines at open; allow initialization and periodically retry.
    connection = serial.Serial(port=None, baudrate=115200, timeout=0.2,
                               write_timeout=2, exclusive=True)
    connection.dtr = False
    connection.rts = False
    connection.port = port
    with output.open("x", encoding="utf-8") as log:
        try:
            connection.open()
            deadline = time.monotonic() + timeout
            next_request = time.monotonic() + 2
            buffered = b""
            started = False
            while time.monotonic() < deadline:
                now = time.monotonic()
                if not started and now >= next_request:
                    connection.write(b"\n/memory\n")
                    next_request = now + 5
                data = connection.read(4096)
                if not data:
                    continue
                text = data.decode("utf-8", errors="replace")
                print(text, end="", flush=True)
                log.write(text)
                log.flush()
                buffered = (buffered + data)[-16384:]
                started = started or b"[memory-begin]" in buffered
                if started and b"[memory-end]" in buffered:
                    print(f"\nSaved device output: {output}")
                    return
            raise SystemExit(f"No complete /memory response in {timeout}s. Log: {output}. "
                             "Check firmware version, port and [READY]; close idf_monitor before using --port.")
        finally:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/esp32ai")
    parser.add_argument("--port", help="Explicit serial port; close idf_monitor first")
    parser.add_argument("--device-only", action="store_true", help="Skip offline build report")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, help="Device log (new file required)")
    args = parser.parse_args()
    if args.timeout <= 0 or (args.device_only and not args.port):
        parser.error("positive timeout required; --device-only requires --port")
    if not args.device_only:
        print_report(build_report(args.build))
    if args.port:
        output = args.output or ROOT / "build/memory-reports" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".txt")
        capture(args.port, args.timeout, output)


if __name__ == "__main__":
    main()
