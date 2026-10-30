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

`source.cache` decides where decoded frames live:

| `cache` | Frames | Use |
|---|---|---|
| `auto` (default) | existing cache, else disk if it has room (+2 GB), RAM if 8 GB stay free, else stream | never fills the disk |
| `disk` | decoded once into `data/.cache/*.npy`, memory mapped (reused next start) | best for speed: needs the decoded size free |
| `ram` | decoded into memory | speed measurements of one file |
| `stream` | decoded while playing, a few dozen frames ahead (low-priority threads) | files larger than disk/RAM |

**Measure speed with `ram` or `disk`.** Streaming decodes gzip (~9 ms and a new 2.6 MB buffer
per 1024 × 1280 frame) inside the simulator, a cost a real camera does not have; it competes
with the workers. Measured with the GenICam data at 250 lines/s: from RAM total 2.4 ms/line,
p99 ~5 ms, 0 lost; streamed the same run varies 4–16 ms/line with p99 up to ~1 s. The info
panel shows each file's storage and the GUI warns *streamed from HDF5: timing includes
decoding*. With ~6 GB free on C: the two GenICam files (5.0 + 5.5 GB decoded) are streamed;
free ~13 GB for the disk cache, or play one file in RAM:
`python gui.py --source data/first.h5` with `source.cache = "ram"`.

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

**Current data: GenICam bin 2, 4 ms** (`data/first.h5`, `data/seconds.h5`, 1024 × 1280 uint16).
Everything comes from their metadata: `raw_inverted` true, m 0.36386 nm/px (bin 1) × 2,
c 842.10 nm, 5 nm bins, dark `dark_b2_e4.0_g1.0_t+21.npz` (chosen by bin/exposure/gain and the
recording's `dark_temperature_c`), Smile `smile_model_bin2.npz`, and the recorded wavelength
range `wavelength_range_bands` = bins 12:171 of 186 → **159 bands 904.65–1695.29 nm**, exactly
like hsi (`LambdaBinner.subset`; without metadata `acquisition.wavelength_range_nm` does the
same). `simulation.json` holds the same values as fall-back. The 4 ms exposure only selects the
preprocessing files (dark etc.); the simulated line rate is separate (`camera.fps`, default 250/s,
changeable live in the GUI). Full chain on this CPU (i9-12900K, 4 workers, frames in RAM): ~2.4 ms per line
at 250/s; maximum ~460 lines/s (6 workers ~500/s), Smile is ~60 % of the time.

**Model: `sgd_first_reflectance.joblib`** (classes mango, pradoo, katom, spectralon, bg; 30 bands
934.7–1079.8 nm; `input_units: reflectance`), trained with
`python train_sgd_coco.py --hdr dataset/first.hdr --labels dataset/first.bip.json --out sgd_first_reflectance.joblib`
on `dataset/first.bip`, which is the hsi reflectance cube of the same scan (it equals this
simulator's chain + Reflectance step: correlation 0.999+). Through the simulator on the
labelled pixels of `first.h5`: 92.0 % (spectralon/bg 100 %, katom 99 %, pradoo 83 %,
mango 33 % - mostly taken for katom: a data/label question, the counts model does the same).
`rf_coco_model.joblib` (your model) was trained on the earlier **counts** version of
`first.bip` (feature std ~11 000): it works with Reflectance **off** (92.3 %) and answers `bg`
everywhere with Reflectance on. `train_sgd_coco.py` now writes `input_units` from the cube
(float ≤ ~5 → reflectance), and the GUI warns when model and Reflectance step disagree.
`model.path: "none"` gives a speed-only stand-in on all bands.

**Reflectance is on at start** (`corrections.reflectance: "auto"`): every source with a
`reflectance/<name>.npy` is calibrated before the model (`first`, `seconds` from
`spectralon-20.h5`). Measured at 250 lines/s, frames from the disk cache: total 2.8 ms per line
(Smile 1.4, reflectance 0.18, model 0.35), latency p50 3.8 / p99 6.5 ms, 0 lines lost.

**Earlier VNIR recordings (Sept 2026, no metadata).** The shipped defaults (bin 2,
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

## Reflectance calibration (last step before inference)

`reflectance = binned counts × factor` with hsi's own code (`modules/radiometric/core.py`,
byte-identical): row-wise factors `(slit rows, bands)` or per-band factors `(bands,)`, loaded
with `load_factor_profile` and checked with `check_factor_bands` (same bands as the binner
produces, by wavelength when the sidecar exists). Output is float32 for the model.

Profiles live in **`reflectance/`**, one per data source, **mapped by name**:

```
data/spectrogram02.h5       -> reflectance/spectrogram02.npy  + spectrogram02.wavelength.npy
dataset/dataset.bip.hdr     -> reflectance/dataset.npy        + dataset.wavelength.npy
```

They are the files hsi writes with `save_factor_profile` (see `reflectance/README.md`).

| `corrections.reflectance` | Effect |
|---|---|
| `none` (default) | a matching `reflectance/<source>.npy` is loaded and offered **off** in the GUI |
| `auto` | on for every source that has its profile; sources without one run without it |
| a path | one profile for all sources; a missing or unfitting file stops the start |

`corrections.reflectance_dir` moves the folder. A profile with other bands, or row-wise factors
with another slit row count, is refused (logged; the switch stays disabled with the reason).
The model artifact may declare `input_units` (`counts` or `reflectance`; `train_sgd_coco.py`
writes `counts`; the current model has none, so `counts` is assumed). When the Reflectance
step does not match, the GUI shows **⚠ model expects …** next to the steps: speed numbers stay
valid, the classes do not. Cost on this CPU: ~0.13 ms per line.

### Making the factors: white reference through the same chain

In operation the camera delivers raw frames and calibration happens in real time, so the
white reference (e.g. Spectralon 40 %) must go through **the same steps as the data** before
it is averaged — otherwise `counts × factor` is not reflectance:

```
white (raw) ─► dark+hot ─► flip ─► Smile ─► binning ─► mean ─► factor = R_ref(λ) / white  ─► reflectance/<data>.npy
data  (raw) ─► dark+hot ─► flip ─► Smile ─► binning ─► × factor (real time) ─► model
```

```bash
# raw white recording (Keep Spectrogram H5), target in slit rows 120..430, row-wise factors
python tools/make_reflectance.py data/white_40.h5 --for run_A run_B \
    --table reflectance/Spectralon_ref-040.xlsx --rows 120:430 --kind row
# binned white cube (ENVI): its dark must come with it
python tools/make_reflectance.py reflectance/spectralon-40.bip.hdr --for run_A \
    --table reflectance/Spectralon_ref-040.xlsx --dark-cube reflectance/dark-40.bip.hdr
```

* The settings (dark, flip, Smile, binning) are the data's: `simulation.json`, the file's
  metadata, `--set` overrides. A raw white without the dark step is refused.
* `--kind band` → `(bands,)` from the ROI mean (a cropped target such as a Spectralon disc);
  `--kind row` → `(slit rows, bands)` with hsi's row window (`--avg-window`, default 20; rows
  outside `--rows` take the nearest ROI rows, as hsi).
* Reference table: hsi's `.xlsx` (nm, R) through `CubicSpline`, as hsi. It must cover the bands.
* The report shows white counts per band, spread across the slit, factor range and bands with
  < 2 % of the peak signal (their factors amplify noise). `--dry-run` writes nothing.
* One white can serve several data sources (`--for A B C`): each gets its own file by name.
* Code: `reflectance_builder.py` (`white_mean`, `reference_reflectance`, `factors_from_white`).

**Current profiles (real): `reflectance/first.npy` and `seconds.npy` from
`reflectance/spectralon-20.h5`** (Spectralon 20 %, GenICam bin 2, 4 ms, same settings as the
data). The file is a slit crop: frames 293:454, sensor rows 453:596 of `data/first.h5` (found
by exact match). The tool puts the crop back at its rows of a 1024-row frame, runs the full
chain (dark `dark_b2_e4.0_g1.0_t+21`, flip, Smile, 159-band binning) and averages the
Spectralon rows 429:570 after the flip over all 161 frames → per-band factors.
Check through the data pipeline itself: on the Spectralon area of `first.h5` the reflectance is
0.2240–0.2603 against the table's 0.2239–0.2605 (mean error +0.02 %, max 0.25 %). Cost ~0.12 ms
per line.

```bash
python tools/make_reflectance.py reflectance/spectralon-20.h5 --for first seconds     --table reflectance/Spectralon_ref-020.xlsx --find-in data/first.h5
```

`--find-in` locates a slit crop in the recording it was cut from (`--slit-offset` /
`--full-rows` give it directly); without either, a cropped raw white cannot meet the full-frame
dark and Smile and is not processed.

**Placeholders until real calibration data exist.** `--placeholder` makes stand-in factors
from any recording (whole cube, raw, no dark required) so the Reflectance step can run now —
its speed is the real one, its values are not. `reflectance/<name>.json` records where every
profile came from and `"placeholder": true`; the log, info panel, step tooltip and the warning
next to the steps say **PLACEHOLDER** while such factors are used. Replace them by running the tool on the real white reference with the same `--for` names
(without `--placeholder`), for example:

```bash
python tools/make_reflectance.py data/spectrogram02.h5 --for spectrogram spectrogram02 \
    --table reflectance/Spectralon_ref-040.xlsx --placeholder
```

## Skipping processing steps (GUI)

The row **Processing steps: Dark + Hot pixel → Flip → Smile → Binning → Reflectance** under the backend
selector switches each step on or off while the simulation runs, e.g. for data that already had
a correction. The change reaches every worker at the next line (nothing is skipped or lost)
and restarts the performance averages; the algorithms are hsi's, only which steps run changes.

| Step | Switchable when | Disabled (greyed) when |
|---|---|---|
| Dark + Hot pixel | a dark reference is loaded: the configured one, or with `corrections.dark: none` an `auto` match (by bin/exposure/gain from the file) offered *off* | no reference fits; hot-pixel repair inside it follows `corrections.hot_pixel` |
| Flip | always for raw input; starts as `raw_inverted` from the file/config | binned input |
| Smile | a model is loaded: the configured one, or with `corrections.smile: none` the `smile_model_bin<N>.npz` that fits the frames, offered *off* | no model fits the frame size |
| Binning | never — always on (the model needs wavelength bins) | always shown checked and disabled |
| Reflectance | `reflectance/<source>.npy` fits the bands (also for binned ENVI input) | no profile for this source, or other bands / slit rows |

For binned input (ENVI with wavelengths) every step except Reflectance is disabled. The tooltip of each box says
why it is on, off or disabled. Headless: `Pipeline.set_steps(dark=False, smile=True)`.

## Fast CPU path (`cpu_fused`)

A new algorithm on the **same calibration data** (Smile `map_x`, dark, wavelength fit are used as
stored, never changed): `fused_cpu.py` does flip + Smile + binning in one Numba pass.

* hsi's CPU Smile (`cv2.remap`, `INTER_CUBIC`) computes a 4 × 4 kernel for every pixel of every
  column and writes the frame before binning. The Smile maps are horizontal only, so 4 taps per
  pixel suffice (as in hsi's CuPy kernel); the fused path uses hsi's fixed-point maps
  (`prepare_maps`) and OpenCV's own weight table (`horizontal_table`), computes only the columns
  the binner uses, and bins on the fly with the binner's `iL`/`iR`/`w`. Dark + hot pixel stay
  hsi's code.
* **Bit-identical** to hsi's CPU path on the GenICam data: 21 frames × dark on/off × 4
  flip/Smile modes, 0 of 27 353 088 binned values differ. Every new configuration is checked
  against hsi at run time (≤ 1 count, hsi's GPU tolerance), otherwise it switches to hsi's path.
* `corrections.backend`: `auto` = GPU if usable, else `cpu_fused`; `cpu` = hsi reference.
  `processing.numba_threads` (per worker) = 6 with 4 workers (24 threads = the CPU's 24).

Full chain, GenICam 1024 × 1280, i9-12900K:

| Setting | Lossless max | At 250 lines/s: p50 / p99 |
|---|---|---|
| hsi CPU, 4 workers, Numba all threads (before) | 465–503/s | 3.5 / 5.5 ms |
| hsi CPU, 4 workers, Numba 1 per worker | 529–562/s | – |
| **cpu_fused, 4 workers × 6 Numba threads (default)** | **714/s** | **3.5 / 6.6 ms** |

## Optics (for later use)

`simulation.json` → `optics`: **slit width 50 µm**, **sensor pixel 5 × 5 µm** (unbinned),
spectrograph magnification 1 (assumed). No processing step uses them yet; the info panel shows
what follows: the slit image covers 10 sensor pixels = 5 pixels at bin 2, i.e. a slit-limited
bandwidth of about 3.64 nm (10 px × 0.3639 nm/px); the 5 nm bins are 1.37 × that. Binned pixel
pitch at bin 2: 10 µm along the slit.

**Belt installation (production):** fore-optics magnification **0.0123**, slit covers **50 cm** of
belt (`optics.lens_magnification`, `optics.slit_fov_mm`):

| Item | Value |
|---|---|
| Pixel on the belt (bin 2) | 10 µm / 0.0123 = **0.813 mm** (unbinned 0.407 mm) |
| Slit width on the belt (along the belt) | 50 µm / 0.0123 = **4.07 mm** - each line sees a 4 mm strip |
| 50 cm field | 500 / 0.813 ≈ **615 pixels** at bin 2 (the slit image is 6.15 mm on the sensor) |
| Square pixels at 24 m/min (0.4 m/s) | 400 / 0.813 ≈ **492 lines/s** (GUI preset "492 ■") |
| One line per slit width at 24 m/min | 400 / 4.07 ≈ 98 lines/s |
| Motion during 4 ms exposure at 24 m/min | 1.6 mm (< the 4.07 mm slit footprint) |

So 24 m/min fits this CPU (full chain max ~714 lines/s) with frames of this size, and better with
only the ~615 slit rows that see the belt. Along the belt the optical resolution is set by the
4 mm slit footprint; sampling at 250-492 lines/s (1.6-0.8 mm per line) oversamples it.

The **lab recording** (240 mm distance) is a different geometry: the 1" Spectralon spans 201
pixels → 0.1264 mm per pixel (`optics.recording_pixel_mm`), recorded at ~87 lines/s; its belt map
is drawn square (`display.line_aspect` 1).

## Belt map proportions

The map is drawn at true pixel proportions: `display.line_aspect` = width of one scan line in
slit pixels (1.0 here: the labelled Spectralon disc is 201 lines × 201 rows), every line drawn
(`canvas_line_stride` 1). The number of lines shown follows the panel's shape, so a round target
stays round at any window size; the newest line is on the right. For other data, measure a round
target: width in lines ÷ height in rows.

## Belt map zones: raw → classified

The belt runs right to left. The right third (`display.raw_zone_fraction`, 0 = off) is the
**RAW** zone: new lines appear there in grey, i.e. the model input (after Dark/Hot → Flip →
Smile → Binning → Reflectance) averaged over bands, with a slowly adapting 1–99 % grey scale. The
left two thirds is the **CLASSIFIED** zone: the same strip of belt continues past the dashed
divider in class colours. Every line is classified as soon as it arrives; the grey zone is a
display choice that shows the input next to the result. Lost lines are magenta in both zones.

The workers compute the grey value (one mean per slit pixel, ~0.05 ms per line, outside the
stage timings), and the aggregator keeps it in a ring beside the class ids. The GUI asks
`snapshot(step=…)` only for the label's resolution, so a refresh copies about 1/4 of the map.

## Throughput on screen

**Camera line rate** (row above the backend selector): a lines/s box plus presets
100 / 250 / 500 / 1000 / 2000 / **Max**, applied at the next line while running (counts and line
order are kept; the stage averages restart). The belt readout shows mm per line at
`camera.belt_speed_m_s`. **Max** = the camera delivers the next line as soon as a buffer is free:
the chain's lossless maximum (measured ~440 lines/s here). The recording's exposure only selects
the preprocessing files and does not limit this rate. Headless: `Pipeline.set_fps(value)`.

A row of cards under the belt map, visible whatever side tab is open: **Camera FPS** (delivered /
target line rate), **Processed FPS** (lines classified per second), **Capacity (est.)** (workers ×
1000 / ms per line, with headroom over the target), **Lost lines** and **Latency p50 / p99**. The
stripe turns orange/red when processing falls behind, headroom drops below 1.3× / 1×, or lines are
lost. The Performance tab has the details per stage.

Measured with the final setup (GenICam 1024 × 1280, full chain incl. reflectance and the
30-band model, frames from the disk cache, i9-12900K CPU):

| Run | Throughput | Lost | Latency p50 / p99 | ms per line per worker |
|---|---|---|---|---|
| 250 lines/s (default line rate) | 250/s | 0 | 3.5 / 5.5 ms | 2.4 |
| maximum, 4 workers (hsi CPU) | **467/s** | 0 | – (backlog test) | 9.1 under full load |
| maximum, 6 workers | 455/s | 0 | – | 13.5 |
| maximum, 8 workers | 157/s | 0 | – | 15.4 (cores oversubscribed) |

So at 250 lines/s the pipeline has ~1.9× headroom; 4 workers is the best setting on
this CPU (more workers compete with OpenCV's own threads). Smile is the largest stage.

## Real-time performance panel

The side panel has two tabs. **Performance** measures speed and efficiency of the pipeline
(not accuracy) — what a real sorter with camera and belt must guarantee:

| Element | Meaning |
|---|---|
| **Time per line** (ms per line per worker) | Dark, Hot pixel, Flip + Smile, Binning, Model, Total — bars for *Now* (last 0.5 s), *Avg* of every backend that ran since Reset (CPU vs CuPy side by side after a switch) and *Peak*. Red dashed line = **budget** = workers × 1000 / line rate: a Total beyond it cannot keep up. |
| **Throughput** (lines/s) | Target, Camera, Processed, Capacity (est.) = workers × 1000 / Total. The estimate is optimistic on CPU (workers share cores); the real maximum: `simulate.py --fps 0 --overrun block`. |
| **Latency** (ms) | Queue wait (time in the buffers before a worker took the line), p50 / p99 / max from sensor time to line counted. A growing queue wait = the chain falls behind. |
| KPIs | Utilisation = processed / capacity, Headroom = capacity / target (< 1 red, < 1.3 orange), Lost lines, Latency p99. |

**Reset** restarts averages and peaks (do it after changing backend, workers or corrections).
At 1600 lines/s with the full chain on this CPU the panel shows the cause directly: Flip + Smile
3.2 ms, Total 4.1 ms over a 2.5 ms budget, capacity ~950/s, queue wait ~270 ms, lines lost.
Chart colours are the `chart_*` tokens of the theme. **Materials** is the former class panel.

## Window layout

The left side is one vertical splitter: **raw spectrogram | belt map | bottom pane**
(backend selector, status line, provenance info).

* Upper handle, or the **Spectrogram** button: shrink or fully collapse the real-time raw
  spectrogram preview. While collapsed it is not rendered at all (saves GUI time at high rates).
* Lower handle: drag down to give the belt map more height; the info panel shrinks to nothing,
  the backend row and status line stay visible.
* Vertical handle between the images and **Material Analytics**, or the **Analytics** button:
  change the panel width or hide it completely; the images get the freed width.
* The layout is remembered between runs (Windows registry,
  `HKCU\Software\NARIT-TSC\PushbroomSortingSimulation`); delete that key to get the default
  1:2 split back.

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
