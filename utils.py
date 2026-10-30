# -*- coding: utf-8 -*-
"""Display helpers."""
import numpy as np

from config import LOST_LINE_RGB, TAB10_RGB_COLORS


def generate_tab10_lut(custom_map: dict[int, list[int]] | None = None) -> np.ndarray:
    """RGB lookup table (256, 3): 0 black, 1..10 Tab10, 255 = lost line (magenta)."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for cid in range(1, 11):
        lut[cid] = TAB10_RGB_COLORS[cid - 1]
    lut[255] = LOST_LINE_RGB
    for cat_id, color_rgb in (custom_map or {}).items():
        if 0 <= cat_id < 255:
            lut[cat_id] = color_rgb
    return lut
