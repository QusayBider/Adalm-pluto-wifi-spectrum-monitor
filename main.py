"""ADALM-PLUTO 2.4 GHz Wi-Fi Spectrum Monitor - Part 1 entry point.

Receive-only, energy-based monitoring of the 2400-2500 MHz band. This file
only provides the menu / command line; every action calls the existing
scripts, which in turn use the modules in src/.

Usage (from the project root, inside the virtual environment):
    python main.py                   interactive menu
    python main.py --mode test       test the Pluto connection
    python main.py --mode sweep      one full-band sweep + figures + .npz
    python main.py --mode analyze    channel analysis of the saved sweep
    python main.py --mode live       live monitoring + heatmap
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

import matplotlib.pyplot as plt

BANNER = """\
============================================
 ADALM-PLUTO 2.4 GHz Wi-Fi Spectrum Monitor
============================================"""

MENU = """
1. Test Pluto connection
2. Single full-band sweep
3. Analyze saved spectrum
4. Start live monitoring
5. Exit"""


def run_test() -> int:
    """Option 1: connect, configure and receive one IQ buffer."""
    from scripts import test_pluto

    return test_pluto.main()


def run_sweep() -> int:
    """Option 2: one 2400-2500 MHz sweep -> full_band_*.png and full_band_spectrum.npz."""
    from scripts import test_full_sweep

    return test_full_sweep.main(verbose=False)


def run_analyze() -> int:
    """Option 3: channel table, full_band_channels.png and channel_occupancy.csv."""
    from scripts import test_channel_detection

    return test_channel_detection.main()


def run_live() -> int:
    """Option 4: repeated sweeps, live spectrum, heatmap, instant + stable candidates."""
    from scripts import live_monitor

    return live_monitor.main()


# Scripts are imported lazily so the menu starts even if pyadi-iio / libiio
# are missing; the import error is then reported for the chosen action only.
ACTIONS: dict[str, tuple[str, Callable[[], int]]] = {
    "test": ("Test Pluto connection", run_test),
    "sweep": ("Single full-band sweep", run_sweep),
    "analyze": ("Analyze saved spectrum", run_analyze),
    "live": ("Start live monitoring", run_live),
}
MENU_CHOICES = {"1": "test", "2": "sweep", "3": "analyze", "4": "live"}


def run_action(mode: str) -> int:
    """Run one action and turn import errors and Ctrl+C into exit codes."""
    title, action = ACTIONS[mode]
    print(f"\n--- {title} ---\n")
    try:
        return action()
    except ImportError as exc:
        print(f"Error: {exc}")
        print("Activate the virtual environment and install requirements.txt (see README).")
        return 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 1
    finally:
        plt.close("all")


def interactive_menu() -> int:
    """Show the menu until the user chooses Exit. Returns the last exit code."""
    print(BANNER)
    last_code = 0
    while True:
        print(MENU)
        try:
            choice = input("\nSelect an option [1-5]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return last_code
        if choice == "5":
            return last_code
        mode = MENU_CHOICES.get(choice)
        if mode is None:
            print("Please enter a number from 1 to 5.")
            continue
        last_code = run_action(mode)
        print(f"\n[{ACTIONS[mode][0]} finished with exit code {last_code}]")


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADALM-PLUTO 2.4 GHz Wi-Fi spectrum monitor (Part 1, receive-only). "
        "Without --mode an interactive menu is shown.",
    )
    parser.add_argument(
        "--mode",
        choices=list(ACTIONS),
        help="run one action directly: test, sweep, analyze or live",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    if arguments.mode is None:
        return interactive_menu()
    return run_action(arguments.mode)


if __name__ == "__main__":
    raise SystemExit(main())
