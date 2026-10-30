# -*- coding: utf-8 -*-
"""Display constants. Simulation settings live in ``simulation.json`` (see ``sim_config.py``)."""

# Matplotlib Tab10 palette (RGB) for class ids 1..10
TAB10_RGB_COLORS = [
    [31,  119, 180],  # Class 1: Tab Blue
    [255, 127, 14],   # Class 2: Tab Orange
    [44,  160, 44],   # Class 3: Tab Green
    [214, 39,  40],   # Class 4: Tab Red
    [148, 103, 189],  # Class 5: Tab Purple
    [140, 86,  75],   # Class 6: Tab Brown
    [227, 119, 194],  # Class 7: Tab Pink
    [127, 127, 127],  # Class 8: Tab Gray
    [188, 189, 34],   # Class 9: Tab Olive
    [23,  190, 207],  # Class 10: Tab Cyan
]

# Belt-map column of a line lost to camera overrun (class id 255)
LOST_LINE_RGB = [255, 0, 255]
