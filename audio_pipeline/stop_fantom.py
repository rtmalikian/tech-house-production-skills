#!/usr/bin/env python3
"""
Send MIDI transport stop and panic messages to a Roland Fantom.

Usage:
    music_venv/bin/python scripts/audio_pipeline/stop_fantom.py
    music_venv/bin/python scripts/audio_pipeline/stop_fantom.py --list-ports
    music_venv/bin/python scripts/audio_pipeline/stop_fantom.py --port "FANTOM-6 7 8"
"""

import argparse
import sys
import time
from typing import Optional

import mido

from fantom_midi_control import list_midi_output_ports


def detect_fantom_port() -> Optional[str]:
    for name in list_midi_output_ports():
        if "FANTOM" in name.upper():
            return name
    return None


def send_stop(port_name: str, panic: bool = True) -> None:
    with mido.open_output(port_name) as port:
        port.send(mido.Message("stop"))

        if panic:
            for channel in range(16):
                port.send(mido.Message("control_change", channel=channel, control=64, value=0))
                port.send(mido.Message("control_change", channel=channel, control=120, value=0))
                port.send(mido.Message("control_change", channel=channel, control=121, value=0))
                port.send(mido.Message("control_change", channel=channel, control=123, value=0))

        time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description="Send MIDI Stop to the Roland Fantom.")
    parser.add_argument("--port", help="Exact MIDI output port name. Defaults to first port containing FANTOM.")
    parser.add_argument("--list-ports", action="store_true", help="List available MIDI output ports and exit.")
    parser.add_argument("--no-panic", action="store_true", help="Only send transport Stop, without all-notes-off messages.")
    args = parser.parse_args()

    if args.list_ports:
        ports = list_midi_output_ports()
        if not ports:
            print("No MIDI output ports found.")
            return 1
        print("MIDI output ports:")
        for name in ports:
            print(f"  {name}")
        return 0

    port_name = args.port or detect_fantom_port()
    if not port_name:
        print("Roland Fantom MIDI output port not found. Use --list-ports, then pass --port.", file=sys.stderr)
        return 1

    send_stop(port_name, panic=not args.no_panic)
    print(f"Sent MIDI Stop to {port_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
