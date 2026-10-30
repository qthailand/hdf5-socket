# Pushbroom sorting simulation (HDF5 / ENVI → classification)

Replays data recorded by the HSI acquisition project (`C:\ClassificationPipeLine\hsi`) as a
high-rate line-scan camera (300 – 1000+ lines/s) and runs the same per-frame processing as
the acquisition, followed by the classifier — to develop industrial sorting software
without camera or belt.

```
raw sensor frame ─► dark ─► hot pixel ─► flip (raw_inverted) ─► Smile ─► lambda binning ─► model ─► belt map / statistics
                    └──────────── modules/preprocessing, dark, spectral_processing (copied from hsi) ┘
```

## Run

```bash
python gui.py                                         # GUI, settings from simulation.json
python gui.py --source dataset/dataset.bip.hdr --fps 300
python simulate.py --fps 1000 --seconds 20            # headless report: throughput, lost lines, latency
python simulate.py --fps 0 --overrun block            # maximum lossless rate of this PC
python simulate.py --set corrections.smile=auto --set corrections.dark=auto
```

`simulate.py --set SECTION.KEY=VALUE` overrides any `simulation.json` value. The socket
streamer `python . send/recv` (`__main__.py`) is a separate tool.

## Input

| Input | Becomes | Stage |
|---|---|---|
| HDF5 `spectrogram` `(frame, slit, spectral)` (HSI *Keep Spectrogram*) | one frame per record | `raw` → full correction + binning |
| ENVI `.hdr` + BSQ / BIL / BIP, `lines`=slit, `samples`=scan | one frame `(slit, bands)` per sample | `binned` if the header has `wavelength`, else `raw` |

`source.envi_scan_axis = "lines"` handles cubes with the scan along lines; big-endian files
and micrometre wavelengths are converted. Frames keep their dtype (uint16 stays uint16).

Compressed HDF5 is decoded once on all cores into `data/.cache/*.npy` and memory mapped
afterwards (1.8 GB of recordings: ~5 s once, then 2 ms to open, 0.26 ms per frame copy);
BSQ/BIL cubes are transposed into the same frame-major cache. `source.cache = "ram"` keeps
them in memory instead. Delete `.cache` at any time.

## Acquisition calibration (no link to the hsi project at run time)

`calibration/` mirrors the hsi layout; copy new files from hsi into the same place:

```
calibration/dark_frames/<backend>/dark_b<bin>_e<exp>_g<gain>[_t<temp>].npz
calibration/smile_maps/<backend>/smile_model_bin<bin>.npz
calibration/wavelength/<backend>/wavelength_bin<bin>.json
```

Each value comes from the recorded file first (`acquisition.use_file_metadata`; HSI writes
`camera_bin`, `exposure_ms`, `gain`, `raw_inverted`, `m_nm_per_px_bin1`, `c_nm`,
`resolution_nm` as HDF5 attributes), otherwise from `simulation.json`. The log and the GUI
list every value with its origin.

* `corrections.dark`: `none` | `auto` (by bin/exposure/gain/temperature, like hsi) | file
* `corrections.smile`: `none` | `auto` (`smile_model_bin<N>.npz`) | file
* `acquisition.wavelength_profile`: `null` (use m/c) | `auto` | file
* binning: `LambdaBinner(m_nm_per_px_bin1 × camera_bin, c_nm, resolution_nm)` exactly as
  `hsi/modules/acquisition/engine.py`.

An `auto` file that is missing or has another frame size is logged and bypassed (as in hsi);
an explicitly named file that does not fit stops the start.

**The recordings in `data/` (Sept 2026) have no metadata.** The shipped defaults (bin 2,
m 0.2903, c 399.791, 2 nm, no flip, no dark, no Smile) reproduce `dataset/dataset.bip` bit
for bit from `spectrogram02.h5` (cube column *j* = frame 2*j*+1), which
`tests/test_simulation.py::test_raw_pipeline_reproduces_the_training_cube` checks.

## Model binding

The model artifact carries `band_wavelengths`. The selected bands are matched to the input
by wavelength (`model.band_tolerance_nm`); a calibration that does not match the training
data stops the start instead of feeding wrong bands. Add the wavelengths to an older
artifact with `python tools/add_band_wavelengths.py model.joblib dataset/dataset.bip.hdr`;
`train_sgd_coco.py` now stores them.

## Pipeline (pipeline.py)

* **Camera**: fixed line clock `camera.fps` (0 = free run), frame copied into one of
  `camera.buffer_frames` pre-allocated buffers. No free buffer = **overrun**: with
  `overrun: "drop"` the line is lost like on a real camera, counted, and drawn magenta on the
  belt map; `"block"` makes the camera wait (throughput test).
* **Workers** (`processing.workers`): parallel correction + binning + classification, each
  with its own `PreprocessingRuntime` (CUDA context per thread). Kernels are warmed up
  before the clock starts.
* **Aggregator**: restores line order and accumulates class pixel counts for **every** line;
  latency = sensor time → line counted (p50/p99/max).
* **GUI** pulls a snapshot at `display.gui_hz`; "real-time" share = all lines since the
  previous refresh. `canvas_line_stride` only thins the drawing.

Measured on this PC (24 threads, CPU backend, 552×804 uint16):

| Chain | Rate | Lost | Latency p50 / p99 |
|---|---|---|---|
| binning + model, 2 workers | 1000/s | 0 | 0.75 / 0.96 ms |
| binning + model, 4 workers | 2000/s | 0 | 0.56 / 0.79 ms |
| binning + model, free run | ~3900/s | 0 | — |
| dark + hot + flip + cubic Smile + binning, free run | ~1100–1300/s | 0 | — |

The cubic Smile remap (1–2 ms/frame on CPU) limits the full chain. Above ~1000 lines/s with
Smile, use the CuPy backend on an NVIDIA GPU (see below). The development PC has no NVIDIA GPU
(Intel UHD 770 only), so the GPU paths are so far checked only by the emulated tests copied
from hsi.

## Same algorithms as hsi

Dark, hot pixel, flip, Smile (loading via hsi `SmileRuntime`) and binning run hsi's own code:
16 files under `modules/` are byte-identical copies, only the package `__init__.py` files are
local. The CPU, CuPy and OpenCV-CUDA backends are hsi's (`modules/preprocessing`), including
its rule that a GPU result is checked once against the CPU reference (at most 1 count) and
falls back to CPU otherwise.

```bash
python tools/sync_hsi.py            # verify against ..\hsi (also a test)
python tools/sync_hsi.py --update   # take over a newer hsi version
```

## Switching CPU / CuPy

* At start: `corrections.backend` in `simulation.json` (`auto`, `cpu`, `cupy_cuda`,
  `opencv_cuda`) or `--backend` for `simulate.py`.
* While running, GUI: the **Correction backend** box above the status line. Backends without
  a usable device are greyed out (tooltip = reason). The panel beside it shows what the
  workers actually run: green stripe = as requested, orange = fallback, with the reason.
* While running, headless: `python simulate.py --backend cpu --switch cupy_cuda@10 --switch cpu@20 --seconds 30`
  compares both in one run with the same data.

A switch takes effect at the next line in every worker: each worker creates a new hsi
`PreprocessingRuntime` in its own thread, warms it up and continues — no line is skipped
(the worker's warm-up time shows up as latency, and as lost lines only if the buffers run
full). hsi's rules still apply: an unavailable or failing GPU backend falls back to the next
candidate and finally CPU, and the first frame of each GPU configuration is checked against
the CPU (max 1 count).

## GPU machine (NVIDIA PC or Jetson Orin)

1. Copy this folder (including `calibration/`, `data/` or your own recordings).
2. `pip install -r requirements.txt`, then CuPy for the installed CUDA major version
   (`nvidia-smi` / JetPack shows it):

   ```bash
   pip install cupy-cuda12x        # CUDA 12.x (driver >= 525; JetPack 6)
   pip install cupy-cuda11x        # CUDA 11.x (JetPack 5)
   ```

   On a Jetson keep the JetPack OpenCV build (it may have CUDA); do not replace it with the
   pip `opencv-python` wheel.
3. Run the check and send back `gpu_report.json`:

   ```bash
   python tools/gpu_check.py
   ```

   It records the device and versions, runs hsi's real-hardware tests (GPU vs CPU, max 1
   count), measures the full chain (dark + hot + flip + Smile + binning + model) at maximum
   lossless rate for CPU and every CUDA backend with 1/2/4 workers (flagging any silent CPU
   fallback), and then fixed rates of 1000–3000 lines/s with lost-line counts and latency.
4. Use the best setting: `corrections.backend = "cupy_cuda"` and its worker count in
   `simulation.json`, or switch live in the GUI.

## Theme

The GUI uses a dark theme by default (`display.theme` in `simulation.json`, or
`python gui.py --theme light`). Styling is separate from the code:

```
themes/app.qss      the stylesheet: shapes, spacing, fonts; colours written as @token
themes/dark.json    token values of the dark theme (bg, surface, text, accent, danger, ...)
themes/light.json   the previous light look as a theme
```

* Recolour: edit the token values in a theme JSON (or copy `dark.json` to `mytheme.json` and
  run `--theme mytheme`).
* Restyle: edit `app.qss`. Widgets are addressed by object name (`#statusLabel`,
  `#classCard`, `#dangerButton`, ...).
* Reusable panel with a coloured left stripe, for any `QFrame`/`QLabel` (also in other
  PyQt5 programs — copy the `themes/` folder):

  ```python
  from themes import apply_theme, make_panel, set_accent, set_accent_color
  apply_theme(app, 'dark')
  make_panel(label, accent='ok')         # ok | warn | error | info | neutral
  set_accent(label, 'warn')              # change it later, e.g. on a state change
  set_accent_color(card, (214, 39, 40))  # stripe in a data colour (needs an objectName)
  ```

  Used for the status line (ok/warn/error), the backend panel, the info panel, the summary
  card and the class cards (stripe = class colour).
* `python -m themes [dark|light|file.json]` opens a gallery of every style with a theme
  switch and a **Reload** button (re-reads `app.qss` and the JSON while you edit them).
* Python sets no colours (a test checks this); only class colours, which are data, are set
  per widget. `tests/test_themes.py` fails if a theme lacks a token used in `app.qss`.
* On Windows the title bar follows the theme (`"dark": true` in the JSON).

## Tests

```bash
python -m pytest -q tests
```

`test_preprocessing*.py` and `cuda_emulation.py` are copied from hsi; `test_simulation.py`
covers sources, calibration resolution, model binding, lossless ordering, overrun counting
the bit-exact reproduction of the training cube, live backend switching and that the
processing code is still identical to hsi; `test_themes.py` checks the themes.
