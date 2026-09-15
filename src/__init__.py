"""ADALM-PLUTO 2.4 GHz Wi-Fi spectrum monitor (Part 1, receive-only).

Modules:
    config            all settings
    pluto_receiver    hardware: connect, configure, receive IQ (no DSP)
    spectrum          FFT, Hann window, linear power, average / max hold, masks
    sweep             LO sweep over 2400-2500 MHz and segment stitching
    channel_detector  noise floor, energy occupancy, center + stable candidates
    heatmap           spectrum / occupancy history and saving
    utils             figure saving and plot-window helpers
"""

__version__ = "0.1.0"
