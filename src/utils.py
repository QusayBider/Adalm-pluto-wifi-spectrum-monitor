"""Shared output helpers for the scripts and main.py.

Project paths, saving matplotlib figures, and showing plot windows without
failing when no GUI is available (e.g. WSL without WSLg/tkinter). No signal
processing and no hardware access.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Project root (the folder containing main.py), independent of the working dir.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Backends that can only write files. Matplotlib silently falls back to "agg"
# when no display (or no GUI toolkit such as tkinter) is available.
NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def is_interactive_backend() -> bool:
    """True if matplotlib can open plot windows."""
    return matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS


def display_path(path: Path) -> Path:
    """``path`` relative to the project root if it lies inside it, else as is."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return Path(path)


def save_figure(figure: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a figure (creating parent folders) and print where it went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    print("Figure saved to:")
    print(display_path(path))


def save_figure_atomic(figure: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save via a temporary file and rename, so viewers never read a partial PNG.

    Used for files that are overwritten while they may be open elsewhere.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=dpi)
    os.replace(temporary, path)


def show_figures() -> None:
    """Open the plot windows if possible. Never raises.

    With WSLg (Windows 11) and python3-tk the windows appear like native apps
    and this call blocks until they are closed. Without a display the PNGs
    have already been saved, so only a note is printed.
    """
    if not is_interactive_backend():
        print(
            f"No interactive matplotlib backend available (using "
            f"'{matplotlib.get_backend().lower()}'); skipping plot windows. "
            "Open the PNGs instead.\n"
            "  For windows under WSLg: sudo apt install python3-tk"
        )
        return

    try:
        plt.show()
    except Exception as exc:
        print(f"Could not display the plot windows ({type(exc).__name__}: {exc}).")
        print("The PNGs were saved successfully.")
