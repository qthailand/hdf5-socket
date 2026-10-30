Smile correction test package (reference row y=552)
==================================================

The supported API in this package is the modular `smile` package. The included
`smile/smile.py` is the original all-in-one implementation and was not synced
with the modular changes; import from `smile` (for example,
`from smile import SmileCorrector`) when testing this delivery.

Contents
--------
- smile/: modular source and tests
- image.png: supplied 16-bit Hg-Ar example image
- model_y552.npz: Smile remap model fitted from image.png with y_ref=552
- wavelength_calibration_6line.json: separate six-line wavelength fit

Install dependencies (in your Python environment)
--------------------------------------------------
pip install numpy opencv-python pytest

Run tests
---------
python -m pytest -q smile

Run the full pipeline benchmark (fit + transform)
-------------------------------------------------
python -m smile.test_cmd image.png --repeat 3

Run production remap benchmark (load once, then 100 transforms)
----------------------------------------------------------------
python -m smile.test_cmd image.png --cal model_y552.npz --production --repeat 100 --backend cpu --interp cubic

The delivered tests passed in the preparation environment: 22 passed, 1
skipped. The skipped test requires optional GUI dependencies.
