# ADALM-PLUTO 2.4 GHz Wi-Fi Spectrum Monitor

## Project Purpose

This is **Part 1** of a university software-defined radio (SDR) project.

The system performs **online, energy-based monitoring of the complete
2400–2500 MHz (2.4 GHz ISM) band** with an ADALM-PLUTO SDR. It measures the
spectrum, determines which standard Wi-Fi channel windows contain RF energy,
estimates which Wi-Fi channel centers best explain the observed broadband
signals, and shows how activity changes over time in a live heatmap.

> **Receive-only.** This software performs **no RF transmission**. The
> ADALM-PLUTO is used exclusively as a receiver; no transmit functionality is
> configured or implemented.

<a href="https://qusaybider.github.io/Adalm-pluto-wifi-spectrum-monitor/" target="_blank">
  🌐 Open the full HTML documentation
</a>

## Features

- Full-band 2400–2500 MHz sensing with 100 % frequency coverage
- Frequency sweeping with 8 overlapping LO steps and fixed manual RX gain
- FFT spectrum (Hann window, DC removal, relative power in dB)
- Average spectrum (20 captures per LO step, averaged in linear power)
- Max Hold spectrum
- DC-artifact and filter-edge exclusion, overlapping segment stitching
- Channel energy analysis for Wi-Fi channels 1–13 (noise-floor based)
- Wi-Fi center candidates (center score, local maxima, non-maximum suppression)
- Temporal stable candidates (center scores averaged over recent sweeps)
- Live spectrum and time-frequency heatmap, updated after every sweep
- Numerical logging (NPZ spectra, CSV occupancy history) with UTC timestamps
- Graceful Ctrl+C: completed sweeps are always saved

## System Architecture

```
ADALM-PLUTO (USB, receive-only)
     |
     v
IQ Samples                      src/pluto_receiver.py
     |
     v
FFT / Power                     src/spectrum.py
     |
     v
Overlapping Frequency Sweep     src/sweep.py
     |
     v
2400–2500 MHz Spectrum          (stitched, linear power + dB)
     |
     +--> Channel Analysis              src/channel_detector.py
     |      energy occupancy + center candidates
     |
     +--> Temporal Candidate Analysis   src/channel_detector.py
     |      stable candidates over the last 5 sweeps
     |
     +--> Heatmap / History             src/heatmap.py
            spectrum_history.npz, occupancy_history.csv
```

| Module | Responsibility |
| --- | --- |
| `main.py` | Menu / command-line entry point; calls the scripts below |
| `src/config.py` | All settings (connection, receiver, sweep, detection, plotting) |
| `src/pluto_receiver.py` | `PlutoReceiver`: connect, configure, receive IQ, verify gain. No DSP |
| `src/spectrum.py` | Hann window, FFT, linear power, average / max hold, valid-bin mask |
| `src/sweep.py` | `SpectrumSweeper` (LO steps) and `stitch_segments()` onto a 5 kHz grid |
| `src/channel_detector.py` | Noise floor, energy occupancy, center candidates, temporal tracker |
| `src/heatmap.py` | `SpectrumHistory`, occupancy records, NPZ/CSV history files |
| `src/utils.py` | Figure saving and plot-window helpers |
| `scripts/` | Stand-alone test and monitoring programs (also used by `main.py`) |

Signal-processing summary:

- **Sweep:** LO centers 2408.5, 2416.5, 2433.5, 2441.5, 2458.5, 2466.5,
  2483.5 and 2491.5 MHz; 20 MS/s, FFT size 16384, 20 captures per center
  after 2 discarded buffers. ±0.2 MHz around each LO and 1 MHz at each capture
  edge are excluded; the paired centers fill each other's excluded regions.
- **Stitching:** valid bins are averaged in linear power into 5 kHz cells over
  2400–2500 MHz; overlapping segments are averaged (max hold: maximum).
- **Noise floor:** 20th percentile of the stitched spectrum.
- **Energy occupancy:** a channel window (center ±10 MHz) is energy-occupied
  if its median is ≥ 5 dB above the noise floor or ≥ 20 % of its bins are
  ≥ 6 dB above it.
- **Center candidate:** `score = max(core median − floor, 0) × core active
  fraction` over center ±8 MHz; local maximum, score ≥ 2.5, 15 MHz separation.
- **Stable candidate:** scores averaged over the last 5 sweeps, same rules,
  supported (score ≥ 2.5) in at least 3 of them.

All thresholds are in `src/config.py`.

## Contents

1. [Setup on a New Computer](#setup-on-a-new-computer) (one-time installation)
2. [Running the Project](#running-the-project)
3. [Generated Output](#generated-output)
4. [Starting the Project Again Later](#starting-the-project-again-later) (daily startup)
5. [Troubleshooting](#troubleshooting)
6. [Important Scientific Limitations](#important-scientific-limitations)
7. [Project Structure](#project-structure)
8. [Quick Start](#quick-start)

---

## Setup on a New Computer

Follow steps A–H once per computer. Nothing is assumed to be installed.
Commands marked **PowerShell (Administrator)** run on Windows; commands marked
**Ubuntu** run inside the WSL Ubuntu terminal.

### A. Windows requirements

You need:

1. Windows 10 or 11 (64-bit)
2. WSL2 (Windows Subsystem for Linux, version 2)
3. Ubuntu installed under WSL
4. usbipd-win (shares USB devices with WSL)
5. An ADALM-PLUTO connected to the computer by USB (use a data cable)

**Install WSL2 and Ubuntu** — PowerShell (Administrator):

```powershell
wsl --install
```

If WSL is already installed, update it instead:

```powershell
wsl --update
```

After a first-time WSL installation **restart Windows**. Then open *Ubuntu*
from the Start menu and create a Linux user name and password when asked.

**Install usbipd-win** — PowerShell (Administrator):

```powershell
winget install --interactive --exact dorssel.usbipd-win
```

Close and reopen PowerShell afterwards so the `usbipd` command is found.
Open PowerShell **as Administrator** (right-click → *Run as administrator*)
whenever you share USB devices with WSL.

### B. Connect the ADALM-PLUTO to WSL

WSL does not see USB devices plugged into Windows automatically; they must be
attached with usbipd.

1. Plug in the ADALM-PLUTO and **start Ubuntu** (keep its terminal open).
2. PowerShell (Administrator) — list USB devices:

   ```powershell
   usbipd list
   ```

   Find the ADALM-PLUTO. Its VID:PID is normally **`0456:b673`**. Note the
   **BUSID** in the first column (for example `2-3`).

3. Share the device (needed once per device and USB port):

   ```powershell
   usbipd bind --busid <BUSID>
   ```

4. Attach it to WSL (Ubuntu must be running):

   ```powershell
   usbipd attach --wsl --busid <BUSID>
   ```

   `<BUSID>` is machine-specific. **Never copy another computer's BUSID** —
   always read it from `usbipd list`. While attached, Windows itself cannot
   use the device.

5. Ubuntu — confirm the device is visible:

   ```bash
   lsusb
   ```

   Expected line (bus and device numbers vary):

   ```
   Bus 001 Device 002: ID 0456:b673 Analog Devices, Inc. LibIIO based AD9363 Software Defined Radio [ADALM-PLUTO]
   ```

   If `lsusb` is not found yet, install the packages in step C first.

### C. Ubuntu system packages

Ubuntu:

```bash
sudo apt update

sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    libiio-utils \
    usbutils
```

| Package | Purpose |
| --- | --- |
| `python3` | Python runtime |
| `python3-pip` | Python package installer |
| `python3-venv` | Virtual environment support |
| `libiio-utils` | libiio library and tools (`iio_info`) to discover and talk to IIO devices such as the Pluto |
| `usbutils` | Provides `lsusb` |

Optional:

```bash
sudo apt install -y git          # only if `git --version` fails (needed in step E)
sudo apt install -y python3-tk   # plot windows through WSLg; PNG files are saved either way
```

### D. Verify the Pluto from WSL

Ubuntu:

```bash
lsusb
iio_info -S usb
```

`iio_info -S usb` should list an ADALM-PLUTO context with its URI in square
brackets, for example:

```
Available contexts:
	0: 0456:b673 (Analog Devices Inc. PlutoSDR (ADALM-PLUTO)), serial=... [usb:1.2.5]
```

The URI (here `usb:1.2.5`) **is different on other computers and can change
after reconnecting the device**. The program connects to the URI configured in
`src/config.py`:

```python
PLUTO_URI = "usb:..."
```

Set it to the URI shown by `iio_info -S usb` on your computer, e.g.
`PLUTO_URI = "usb:1.2.5"`. (`PLUTO_URI = "usb:"` selects the only connected
USB Pluto automatically and does not need updating after reconnects.)

If the Pluto appears only with `sudo iio_info -S usb`, see
[Troubleshooting](#troubleshooting), case 3.

### E. Clone the repository

Ubuntu — keep the project inside the Linux file system (not under `/mnt/c`):

```bash
cd ~
git clone <REPOSITORY_URL>
cd WiFi_Channel_Sensing
```

Replace `<REPOSITORY_URL>` with the URL of this repository.

### F. Create the Python virtual environment

Ubuntu, inside `~/WiFi_Channel_Sensing`:

```bash
python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

After activation the prompt normally begins with **`(.venv)`**. `.venv` is not
part of the repository, so it must be created on every new computer.
`requirements.txt` installs `pyadi-iio` (which also installs the `pylibiio`
bindings, the `iio` module), `numpy` and `matplotlib`.

In VS Code (WSL extension) select `.venv/bin/python` as the interpreter.

### G. Verify the Python dependencies

Ubuntu, with `(.venv)` active:

```bash
python -c "import iio; print('iio OK')"

python -c "import adi; print('pyadi-iio OK')"
```

If both print `OK`, the Python environment is ready.

### H. First hardware test

Before starting the main program, test the connection:

```bash
python -m scripts.test_pluto
```

Expected type of result (sample values differ every time):

```
Connecting to ADALM-PLUTO...
URI: usb:...
Connected successfully.
Configured: LO 2437.000 MHz, 20.0 MS/s, buffer 16384
Samples received: 16384
First 5 IQ samples:
[...]
```

**If this fails, do not continue** to spectrum monitoring — fix the connection
first (see [Troubleshooting](#troubleshooting)).

---

## Running the Project

Always run from the project root with the virtual environment active.

### Main program

```bash
python main.py
```

```
============================================
 ADALM-PLUTO 2.4 GHz Wi-Fi Spectrum Monitor
============================================

1. Test Pluto connection
2. Single full-band sweep
3. Analyze saved spectrum
4. Start live monitoring
5. Exit
```

| Option | What it does | Needs the Pluto |
| --- | --- | --- |
| 1. Test Pluto connection | Connects and receives one IQ buffer | yes |
| 2. Single full-band sweep | One 2400–2500 MHz sweep; saves `full_band_*.png` and `full_band_spectrum.npz` | yes |
| 3. Analyze saved spectrum | Channel table, `full_band_channels.png`, `channel_occupancy.csv` from the last sweep (run option 2 first) | no |
| 4. Start live monitoring | 30 repeated sweeps: live spectrum, heatmap, instant and stable candidates, history files. **Ctrl+C** stops early and saves | yes |
| 5. Exit | | |

The same actions without the menu:

```bash
python main.py --mode test
python main.py --mode sweep
python main.py --mode analyze
python main.py --mode live
```

During live monitoring each sweep prints a short block:

```
Sweep 5/30   (1.09 s, spectrum + heatmap updated)
  Noise floor:       -38.3 dB
  Energy windows:    [4, 5, 6, 7, 8, 9]
  Instant candidate: [7]
  Stable candidate:  [6]   (last 5)
```

The number of sweeps and all other settings are in `src/config.py`.

### Developer commands

Stand-alone scripts (the menu uses the same code):

```bash
python -m scripts.test_pluto               # connection and IQ samples
python -m scripts.test_spectrum            # channel 6: FFT, average, max hold, masks
python -m scripts.test_full_sweep          # full-band sweep with detailed log
python -m scripts.test_channel_detection   # channel analysis of the saved sweep
python -m scripts.live_monitor             # live monitoring
```

---

## Generated Output

The programs create these files at runtime. They are ignored by Git
(`.gitignore`), so a freshly cloned repository contains only empty folders.

**Figures — `results/figures/`**

| File | Created by | Content |
| --- | --- | --- |
| `live_spectrum.png` | live | Latest stitched spectrum, noise floor, instant (dashed orange) and stable (solid purple) candidates; overwritten after every sweep |
| `live_heatmap.png` | live | Time-frequency heatmap (frequency left→right, oldest sweep at top, color = relative power in dB); overwritten after every sweep |
| `full_band_spectrum.png` | sweep | Stitched average spectrum 2400–2500 MHz |
| `full_band_average_maxhold.png` | sweep | Stitched average and max hold |
| `full_band_segments.png` | sweep | Valid part of each LO segment before stitching (diagnostic) |
| `full_band_channels.png` | analyze | Spectrum, noise floor, energy status per channel, center candidates |
| `channel6_*.png` | `test_spectrum` | Single-channel diagnostic figures |

**Processed data — `data/processed/`**

| File | Created by | Content |
| --- | --- | --- |
| `full_band_spectrum.npz` | sweep | Frequencies, stitched average and max hold (linear and dB), capture settings |
| `channel_occupancy.csv` | analyze | One row per channel: powers, excess, active fraction, energy status, center score, candidate flag |
| `spectrum_history.npz` | live | `power_history_db` (sweeps × 20000 bins), UTC timestamps, noise floor per sweep, per-channel matrices, settings |
| `occupancy_history.csv` | live | One row per sweep: timestamp, noise floor, energy-occupied channels, instant and stable candidates |

Without a plot window (normal under WSL without `python3-tk`), open the PNGs
from Windows, e.g. `explorer.exe results/figures`. The live PNG and CSV files
are rewritten after every sweep and can be viewed while monitoring runs.

---

## Starting the Project Again Later

After rebooting Windows, restarting WSL (`wsl --shutdown`) or unplugging the
ADALM-PLUTO, the device must be attached to WSL again.

1. Start Ubuntu.
2. PowerShell (Administrator):

   ```powershell
   usbipd list
   usbipd attach --wsl --busid <BUSID>
   ```

   If the device is listed as *Not shared* (e.g. after using a different USB
   port), bind it first:

   ```powershell
   usbipd bind --busid <BUSID>
   ```

3. Ubuntu:

   ```bash
   lsusb
   iio_info -S usb
   ```

   If the URI in square brackets changed, update `PLUTO_URI` in
   `src/config.py`.

4. Start the program:

   ```bash
   cd ~/WiFi_Channel_Sensing
   source .venv/bin/activate
   python main.py
   ```

---

## Troubleshooting

Work through the checks in order: `lsusb` → `iio_info -S usb` → Python
imports → `python -m scripts.test_pluto`. The first failing check shows
where the problem is.

**1. `usbipd` command not found (PowerShell)**
usbipd-win is not installed, or PowerShell was opened before installing it.
Install it (step A) and open a new PowerShell as Administrator.

**2. `lsusb` does not show the ADALM-PLUTO**
The USB device is not attached to WSL. In PowerShell (Administrator):

```powershell
usbipd list
usbipd attach --wsl --busid <BUSID>
```

The device must show *Attached*. Ubuntu must be running during `attach`. If
it is still missing, try another USB port or cable, then `bind` and `attach`
again.

**3. `lsusb` shows the Pluto but `iio_info -S usb` does not**
Check the libiio installation and USB access:

```bash
sudo apt install libiio-utils
```

If `sudo iio_info -S usb` works but `iio_info -S usb` does not, your user has
no permission to open the device. Add a udev rule, then detach and attach the
device again with usbipd:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0456", ATTRS{idProduct}=="b673", MODE="0666"' \
    | sudo tee /etc/udev/rules/53-adi-plutosdr-usb.rules
sudo udevadm control --reload-rules
```

**4. `import iio` fails**
The virtual environment is not active or the requirements are not installed.
Check that the prompt starts with `(.venv)`; otherwise run
`source .venv/bin/activate` in the project folder, then
`python -m pip install -r requirements.txt`. If the error mentions `libiio`,
install `libiio-utils` (step C).

**5. `import adi` fails**

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**6. Connection to the Pluto URI fails** (e.g. "Could not connect ... No device found")
The configured URI does not match the device. Run `iio_info -S usb` and set
`PLUTO_URI` in `src/config.py` to the URI shown in square brackets.

**7. Matplotlib does not open a window**
Not an error under WSL: all figures are still saved as PNG files in
`results/figures/`. For optional plot windows through WSLg:

```bash
sudo apt install python3-tk
```

**8. Permission or connection problems after reconnecting the Pluto**
Detach and re-attach it from PowerShell (Administrator), then check again:

```powershell
usbipd detach --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

```bash
lsusb
iio_info -S usb
```

Update `PLUTO_URI` if the URI changed. A "Sweep failed ... No such device"
message during monitoring means the Pluto was disconnected; completed sweeps
are still saved.

---

## Important Scientific Limitations

1. **Relative power only.** Power is shown in relative dB, not calibrated
   dBm. The ADALM-PLUTO is not amplitude-calibrated; only differences
   between frequencies and sweeps with the same settings are meaningful.
2. **No simultaneous observation.** The ADALM-PLUTO does not observe the
   entire 100 MHz band simultaneously; one capture covers about 20 MHz.
3. **Sequential sweeps.** The 2400–2500 MHz band is measured using sequential,
   overlapping LO sweeps (about 1.1 s per sweep) that are stitched together.
   Different frequencies within one spectrum or heatmap row were measured at
   slightly different times.
4. **Bursty traffic.** Wi-Fi is bursty, so RF activity may change during a
   single sweep, and overlapping segments may show different levels.
5. **Energy-based detection.** Occupancy is detected from RF energy only.
   Bluetooth, microwave ovens, other ISM devices or receiver spurs in the same
   frequency range are detected in the same way. Because 2.4 GHz channel
   windows overlap (5 MHz spacing, about 20 MHz width), one signal can make
   several neighbouring channel windows energy-occupied.
6. **No packet decoding.** No IEEE 802.11 packets are decoded.
7. **Center candidates are interpretations.** A center candidate means that
   the broadband energy is consistent with a signal centered near a standard
   Wi-Fi channel frequency, not that an access point or its identity has been
   proven. Because 2.4 GHz channels overlap, networks closer than about
   15 MHz may merge into one candidate, and energy centered between two
   channels can make the candidate alternate; temporal averaging reduces but
   does not remove this.
8. **Validation pending.** Controlled validation using a router fixed to known
   channels has not yet been performed and remains an optional future
   validation experiment. Detection thresholds are initial experimental
   values.

---

## Project Structure

```
WiFi_Channel_Sensing/
│
├── main.py                       Entry point (menu and --mode)
├── README.md
├── requirements.txt              pyadi-iio, numpy, matplotlib
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py                 All settings (including PLUTO_URI)
│   ├── pluto_receiver.py         Hardware access (receive-only)
│   ├── spectrum.py               FFT / power / average / max hold / masks
│   ├── sweep.py                  LO sweep and stitching
│   ├── channel_detector.py       Energy occupancy, center and stable candidates
│   ├── heatmap.py                Spectrum and occupancy history
│   └── utils.py                  Figure helpers
│
├── scripts/
│   ├── __init__.py               Makes `python -m scripts.<name>` reliable
│   ├── test_pluto.py
│   ├── test_spectrum.py
│   ├── test_full_sweep.py
│   ├── test_channel_detection.py
│   └── live_monitor.py
│
├── data/
│   ├── raw/
│   └── processed/                NPZ and CSV results (generated)
│
└── results/
    ├── figures/                  PNG figures (generated)
    └── logs/
```

---

## Quick Start

For a computer where [Setup on a New Computer](#setup-on-a-new-computer) has
already been completed.

**Windows PowerShell (Administrator):**

```powershell
usbipd list
usbipd bind --busid <BUSID>          # only if the Pluto is not shared yet
usbipd attach --wsl --busid <BUSID>
```

**Ubuntu:**

```bash
lsusb
iio_info -S usb                      # update PLUTO_URI in src/config.py if the URI changed

cd ~/WiFi_Channel_Sensing
source .venv/bin/activate

python -m scripts.test_pluto
python main.py
```

**Freshly cloned repository?** `.venv` does not exist yet. Create it once
before `source .venv/bin/activate`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
