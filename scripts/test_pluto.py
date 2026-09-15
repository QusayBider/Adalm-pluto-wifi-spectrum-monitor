"""Step 1 smoke test: connect to the ADALM-PLUTO and read one IQ buffer.

Run from the project root:
    python -m scripts.test_pluto
"""

from __future__ import annotations

import sys
from pathlib import Path

# Also allow "python scripts/test_pluto.py" from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.pluto_receiver import PlutoConnectionError, PlutoReceiver

TROUBLESHOOTING = """\
Unable to connect to ADALM-PLUTO.
Check:
- usbipd device attachment (Windows: usbipd list -> state "Attached")
- lsusb                    (should list 0456:b673 Analog Devices PlutoSDR)
- iio_info -S usb          (should list the Pluto and its URI in [brackets])
- configured PLUTO_URI     (src/config.py, currently '{uri}')
See README.md -> Troubleshooting."""


def main() -> int:
    """Connect, configure, receive one buffer and print the first IQ samples.

    Returns:
        Process exit code (0 on success, 1 on any connection/receive error).
    """
    print("Connecting to ADALM-PLUTO...")
    print(f"URI: {config.PLUTO_URI}")

    receiver = PlutoReceiver(uri=config.PLUTO_URI)
    try:
        receiver.connect()
        print("Connected successfully.")

        receiver.configure()
        print(
            f"Configured: LO {receiver.center_freq / 1e6:.3f} MHz, "
            f"{receiver.sample_rate / 1e6:.1f} MS/s, "
            f"buffer {receiver.buffer_size}"
        )

        samples = receiver.receive_samples()
    except PlutoConnectionError as exc:
        print()
        if not receiver.is_connected:
            print(TROUBLESHOOTING.format(uri=config.PLUTO_URI))
            print()
        print(f"Error: {exc}")
        return 1
    finally:
        receiver.close()

    print(f"Samples received: {samples.size}")
    print("First 5 IQ samples:")
    print(samples[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
