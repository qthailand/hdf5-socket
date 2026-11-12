# Pushbroom sorting simulation (HDF5 / ENVI → classification)

Replays data recorded by the HSI acquisition project (`C:\ClassificationPipeLine\hsi`) as a
high-rate line-scan camera (300 – 1000+ lines/s) and runs the same per-frame processing as
the acquisition, followed by the classifier — to develop industrial sorting software
without camera or belt. The same pipeline also runs on a **live camera** (hsi's camera
backends, `source.kind = "camera"`, see *Live camera*).

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
python gui.py --live                                  # live camera instead of files
python simulate.py --live --seconds 30 --set acquisition.exposure_ms=2
```

`--set SECTION.KEY=VALUE` (`gui.py` and `simulate.py`) overrides any `simulation.json` value
(JSON values; `null` clears one).

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

## Live camera (`source.kind = "camera"`)

The same pipeline reads a real camera instead of files: `python gui.py --live`,
`python simulate.py --live`, or `"source": {"kind": "camera"}` in `simulation.json`. Everything
after the frame buffers is unchanged (corrections, model, belt map, statistics), because a camera
delivers what a *Keep Spectrogram* recording stores: the raw `uint16` frame `(slit, spectral)` at
the hardware bin.

The camera code is hsi's `modules/camera` (POA, GenICam, XIMEA), byte-identical copies like the
processing code (see *Same algorithms as hsi*). The vendor SDK folders (`modules/camera/poa`,
`modules/camera/ximea`: ctypes wrappers and DLLs, the XIMEA one 110 MB) are not in git, as in
hsi; copy them once:

```bash
python tools/sync_hsi.py --update --only camera       # camera files incl. the poa/ and ximea/ SDKs
python -m modules.camera --backend genicam --cti C:\path\producer.cti --bin 2   # hsi's hardware check
```

GenICam also needs `pip install harvesters` and a GenTL producer (`.cti`); POA and XIMEA ship
Windows DLLs only.

At start the camera gets the `acquisition` values, which then also select the dark, Smile and
wavelength files, as the attributes of a recording do (origin `camera:...` in the info panel):

| Key | Live meaning |
|---|---|
| `acquisition.camera_backend` | `poa`, `genicam` or `ximea`; also the `calibration/*/<backend>/` folder for a camera without a profile |
| `acquisition.camera_bin` | hardware bin set on the camera (an unsupported bin is an error) |
| `acquisition.exposure_ms`, `gain` | required; applied to the camera, and the values it applied pick the dark |
| `acquisition.raw_inverted` | slit flip, as for files |
| `acquisition.wavelength_profile` | `auto`: `wavelength_bin<bin>.json` (camera profile, else `calibration/wavelength/<backend>/`) is **required** (a camera carries no m/c of its own; starting without it is an error rather than another camera's axis); `"none"` uses `m_nm_per_px_bin1` / `c_nm` on purpose |
| `live.camera_index` | device index (GenICam, XIMEA; POA opens its first camera) |
| `live.cti_path` | GenICam GenTL producer; `null` = the SVS producer (`sv_gev_tl_x64.cti`, `libsv_gev_tl_x64.cti`) in one of the `GENICAM_GENTL64_PATH` folders; a missing file is an error naming the path |
| `live.sensor_pixel_size` | GenICam `SensorPixelSize` 8/10/12 bits: must be what the darks were recorded with (it changes the value scaling) |
| `live.read_timeout_ms` | longest wait for one frame, 1–10000 (default 100); a stop waits for one read plus 1 s |
| `live.reader` | `process` (default): the camera is opened and read in a process of its own, writing into shared-memory frame buffers, so the GUI and the workers cannot slow the camera through the GIL (`camera_process.py`); `thread`: read by a thread of this process (the tests' fake cameras) |

The sensor temperature at start picks the nearest dark, as in hsi. Reflectance `auto` looks for
`reflectance/live_<backend>.npy`. `resolution_nm` comes from the config for a live camera (a
recording carries its own).

How it runs (`Pipeline._live_loop`, `camera_source.py`):

* The camera **free-runs** at the rate its exposure and readout allow (hsi sets no frame rate
  either). `camera.fps`, `overrun: "block"` and `loop` apply to files only; the line-rate box
  shows the measured rate (frames the camera delivers per second), read-only, and the mm/line
  readout uses it. When the camera stops sending, the rates fall to 0, the box shows *no frames*,
  the status line turns orange and counts the read timeouts.
* Each frame is written straight into a free buffer (GenICam converts into it, the other
  backends copy). With no free buffer the frame is still read and, unless a buffer came free
  during the read, dropped as an **overrun**; frames the camera or the link lost (GenICam
  frame-id gaps, POA's dropped counter) are lost lines too (`lines_lost_camera`). Both are drawn
  magenta. XIMEA reports no drops.
* **Latency** starts at the camera timestamp for GenICam (camera clock latched to the host and
  re-latched every 30 s; implausible stamps trigger one immediate re-latch, then fall back to
  the arrival time), otherwise at the frame's arrival (`Snapshot.line_clock`: `camera` /
  `arrival`). *Queue wait* in the Performance tab is always the time in the buffers.
* Gain, exposure and bin stay fixed while running, because they select the calibration: change
  them in `simulation.json` or with `--set` and start again. The pipeline's camera thread owns the
  camera and closes it when the pipeline stops. If a read hangs inside the SDK, closing the
  window a second time quits anyway (the camera is then not closed; the OS releases it).
* `simulate.py --live` counts `--seconds` from the start of the stream, reports the camera's
  mean rate, and exits with 1 when no frame arrived.
* To focus the lens, use the **FOCUS** card beside the preview with `gui.py --live`: it scores
  the camera's raw frames as they arrive (see *Focusing the camera*).

## Acquisition calibration (no link to the hsi project at run time)

Calibration belongs to one physical camera, so each camera has a **profile** named after the
model and serial it reports (a live camera, or the `camera_model` / `camera_serial` attributes
hsi writes into a recording):

```
profiles/<model>-<serial>/                 e.g. profiles/Apollo-M-MAX-CAMP7161A470A1209000/
  profile.json                             backend, model, serial
  dark/dark_b<bin>_e<exp>_g<gain>[_t<temp>].npz
  smile/smile_model_bin<bin>.npz           (+ history/)
  wavelength/wavelength_bin<bin>.json      (+ resolution_bin<bin>.json, history/)
  reflectance/refl_b<bin>_g<gain>.npy      (+ .wavelength.npy, .json, history/)
```

When a profile has a folder (`dark/`, `smile/`, `wavelength/`), only that folder is searched,
so files of another camera of the same backend are never used. A camera without model/serial,
or a profile without that folder, uses the hsi layout per backend (GenICam and XIMEA files are
still there):

```
calibration/dark_frames/<backend>/dark_b<bin>_e<exp>_g<gain>[_t<temp>].npz
calibration/smile_maps/<backend>/smile_model_bin<bin>.npz
calibration/wavelength/<backend>/wavelength_bin<bin>.json
```

`tools/migrate_profiles.py --from-recording <a recording of the camera> [--apply]` moves one
backend's files into a camera's profile, including its old `reflectance/live_<backend>.npy` as
`refl_b<bin>_g<gain>.npy`. Without `--apply` it only prints the plan. The calibration files
carry no serial, so only move them for the camera they were made with. The POA Apollo-M MAX
(CAMP7161A470A1209000) was moved on 2026-11-10. `corrections.profiles_dir` moves the folder.

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

**Model: `models/sgd_first_reflectance.joblib`** (classes mango, pradoo, katom, spectralon, bg; 30 bands
934.7–1079.8 nm; `input_units: reflectance`), trained with
`python tools/train_sgd_coco.py --hdr dataset/first.hdr --labels dataset/first.bip.json --out models/sgd_first_reflectance.joblib`
on `dataset/first.bip`, which is the hsi reflectance cube of the same scan (it equals this
simulator's chain + Reflectance step: correlation 0.999+). Through the simulator on the
labelled pixels of `first.h5`: 92.0 % (spectralon/bg 100 %, katom 99 %, pradoo 83 %,
mango 33 % - mostly taken for katom: a data/label question, the counts model does the same).
`models/rf_coco_model.joblib` (your model) was trained on the earlier **counts** version of
`first.bip` (feature std ~11 000): it works with Reflectance **off** (92.3 %) and answers `bg`
everywhere with Reflectance on. `tools/train_sgd_coco.py` now writes `input_units` from the cube
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
by wavelength (`model.band_tolerance_nm`), so wrong bands are never fed to the model.

**After a new wavelength calibration** the camera's bands lie a little beside the bands an older
model was trained on. Such a model stays usable: the input is interpolated onto the model's
bands (linear, by wavelength; `model.resample`, default `true`). The binding line then reads
`resampled to the model's bands ... nearest input band up to 1.17 nm away`, and the model's tile
in the GUI shows `resampled 1.2 nm` in the warning colour. It applies when the input covers the
model's bands (the first or last may lie up to half a band outside: it is extrapolated) and has the training data's band spacing (0.8-1.25 times); another spectral
resolution or a range that does not cover the model is still refused, with the reason.
The cost is about 0.1 ms per line for a package model and nothing for a linear one (the
interpolation is folded into its weights). Interpolating assumes the new calibration describes
the same spectra more accurately; check a resampled model on known material, and retrain it when
the result matters. `model.resample: false` refuses such models as before. Add the wavelengths to an older
artifact with `python tools/add_band_wavelengths.py model.joblib dataset/dataset.bip.hdr`;
`tools/train_sgd_coco.py` now stores them.

## Models trained in HSI MLOps (`.hsipkg`)

The classification models of this project are trained elsewhere (HSI MLOps, `hsi_lab_core`) and
delivered as a package: `*.hsipkg`, usually inside a `.zip`. `model.path` takes either file
directly; nothing has to be converted and `hsi_lab_core` is not needed:

```bash
python simulate.py --set model.path=models/Kratom_adulteration_1794300998.hsipkg
python hsi_package.py models/*.hsipkg        # what a package holds, and whether it is accepted
```

**Switching the model while running.** Put the model files (`.joblib`, `.hsipkg`, or the `.zip`
around a package) into the `models` folder. The **Materials** tab shows one tile per file of that
folder only (`model.folders`, default `["models"]`) below the class list; ↻ looks again. The
tile of the model in use is highlighted. Clicking another tile switches at the next camera
line: nothing stops, no line is lost, and the class statistics and the belt map start again. A
model that does not fit the input (other bands, another calibration) is shown greyed out with
the reason in its tooltip. `model.path` is still the model loaded at start; one
outside `models` runs but is not offered in the list.

The model runs as its training software runs it (`inference_engine.HsiPackageClassifier`):
reflectance → SNV per pixel over **all** training bands → the package's selected bands → its
StandardScaler → Gaussian maximum likelihood (MLC, per class a mean and a full covariance).
What makes that trustworthy without the training software:

* `hsi_package.py` reads the arrays out of the package's joblib pickles with a restricted
  unpickler (numpy arrays and attribute holders only: no code of the package runs) and checks
  the manifest's `sha256` of every file it reads.
* The package carries the class means of its training spectra after preprocessing
  (`training_spectral_reference.json`). The loader passes them through its own band selection
  and scaler and requires the model's class means to come out (they do, to 1e-14, for both
  delivered packages) and the SNV convention to match (population SD over all bands).
* A package using a step that is not implemented (another preprocessing, PCA, spectral
  indices, resampling, another algorithm) is refused by name. Implemented today: SNV, band
  selection, StandardScaler, MLC with full covariances.
* The classes are the arg max of float64 `log prior + Gaussian log-density`; `predict` computes
  it in float32 and stops a class as soon as it has lost (same result: 61 440 pixels of
  `first.bip` agree with a float64 `scipy` reference, and `tests/test_hsi_package.py` checks it).

Things to know:

* **Bands.** SNV needs every training band, so the input must supply all 159 (not only the
  137 the model uses) within `model.band_tolerance_nm`. `Kratom_adulteration` was trained on
  this camera's grid (904.65–1695.29 nm, offset 0.005 nm). `MLC_874171` was trained on another
  wavelength calibration (905.06–1694.12 nm: up to 1.17 nm off): it runs on the input resampled
  onto its bands (see *Model binding*).
* **Input.** The package declares reflectance, so the Reflectance step must be on (the GUI
  warns otherwise). SNV removes brightness and offset per pixel; a wrong white reference still
  changes the spectral shape.
* **Classes and colours** come from the package (ids from 0; `Background` is hidden in the
  shares by `display.hidden_classes`). A pixel with a constant or non-finite spectrum (dead
  pixel, the all-zero warm-up frame) has no SNV and gets the package's background class.
* **Speed** (i9-12900K, slit 1024, frames in RAM): model 0.9–1.1 ms per line against 0.3 ms for
  the linear model; the full chain still reaches ~730 lines/s with 4–6 workers.
* **Not verified here:** the result was not compared with `hsi_lab_core` itself (it is not on
  this machine). The check that remains is one cube classified in HSI MLOps and here.
  `predict_confidence` is the Gaussian posterior, which the package itself calls uncalibrated;
  `tools/check_calibration.py` still expects a joblib artifact.

## Confidence and calibration (offline)

The running system classifies with `predict` only (the class, no probability), so confidence
costs nothing at run time. To check a model offline,
`PushbroomLineClassifier.predict_confidence` also returns the **confidence** of each pixel: the
probability of the winning class exactly as sklearn's `predict_proba` gives it for the
`log_loss` SGD model (one-vs-rest: sigmoid of every class score, normalised to sum 1; checked
against sklearn in the tests), computed from the scores divided by the artifact's
`temperature`. The class is always the same as `predict`'s.

`tools/check_calibration.py` checks that "90 %" means right about 90 % of the time: it runs the
artifact through `PushbroomLineClassifier.predict_confidence` on the labelled pixels of the
training cube (`training_header`, COCO polygons `<cube>.json`) and prints a reliability table
(per confidence bin: pixels, mean confidence, actual accuracy), the ECE, the accuracy below and
above `--min-confidence` and per class accuracy, precision and mean confidence. By default it
uses the 20 % held-out pixels of `tools/train_sgd_coco.py` (the split depends only on the labels, so
it is reproduced exactly; `--split train|all`, `--temperature 1` for the uncalibrated model).

```bash
python tools/check_calibration.py                       # models/sgd_first_reflectance.joblib
```

The plain SGD model is **under-confident**: on the held-out pixels of `first` it is right 92.0 %
of the time but reports 77.4 % on average (pixels reported at 80-90 % are right 99 % of the
time; ECE 0.146). `tools/train_sgd_coco.py` therefore fits a **temperature** T on the held-out pixels
(the one scalar that minimises their log loss) and stores it in the artifact;
`predict_confidence` divides the scores by T before the sigmoid (T < 1 sharpens). The class
(`predict`, argmax) does not change. For `first` T = 0.474: mean confidence 90.4 % for 92.0 %
accuracy, ECE 0.016 (0.015-0.017 when T is fitted on one half of the held-out pixels and
checked on the other). Artifacts without `temperature` keep T = 1.0 (sklearn's value); retrain
to add it (`python tools/train_sgd_coco.py --hdr dataset/first.hdr --out models/sgd_first_reflectance.joblib`
gives the same weights).

* One T fixes the overall scale, not every class: mango pixels assigned to mango are right 98 %
  of the time at 50 % confidence, katom 80 % at 87 % (two thirds of the mango pixels are taken for
  katom: mango recall 33 %). That is the model, not the calibration.
* The held-out pixels are neighbours of training pixels in the same recording. On a new
  recording (other fruit, lighting, belt) accuracy is lower; label part of it and run the check
  on it (`--hdr new.hdr --labels new.json`) before trusting its confidence there.

## Pipeline (pipeline.py)

* **Camera**: fixed line clock `camera.fps` (0 = free run), frame copied into one of
  `camera.buffer_frames` pre-allocated buffers. No free buffer = **overrun**: with
  `overrun: "drop"` the line is lost like on a real camera, counted, and drawn magenta on the
  belt map; `"block"` makes the camera wait (throughput test). A live camera
  (`source.kind = "camera"`) fills the same buffers at its own rate and always drops on overrun
  (see *Live camera*).
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

Dark, hot pixel, flip, Smile (loading via hsi `SmileRuntime`) and binning run hsi's own code,
and the live camera hsi's camera backends: the files listed in `tools/sync_hsi.py` are
byte-identical copies, only the package `__init__.py` files are local. The CPU, CuPy and OpenCV-CUDA backends are hsi's (`modules/preprocessing`), including
its rule that a GPU result is checked once against the CPU reference (at most 1 count) and
falls back to CPU otherwise.

```bash
python tools/sync_hsi.py            # verify against ..\hsi (also a test)
python tools/sync_hsi.py --update   # take over a newer hsi version
python tools/sync_hsi.py --update --only camera   # only modules/camera (and its SDK folders)
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
The model artifact may declare `input_units` (`counts` or `reflectance`; `tools/train_sgd_coco.py`
writes `counts`; the current model has none, so `counts` is assumed). When the Reflectance
step does not match, the GUI shows **⚠ model expects …** next to the steps: speed numbers stay
valid, the classes do not. Cost on this CPU: ~0.13 ms per line.

### Calibrating in the GUI: draw the white standard on RAW

Until a holder keeps the Spectralon under the slit, calibrate on the running view:

1. **Sensor…** (live camera): exposure, gain and bin. The dialog lists the stored darks and
   shows which dark the values will use (without one the Dark step is off, so the
   reflectance would be wrong). **Apply** restarts the pipeline: the camera is reopened with
   the new values, and the dark, Smile and wavelength files are picked again. **Save to
   simulation.json** keeps the values for later runs.
2. **Reflectance…**: pick the white standard (a table in `reflectance/`, e.g.
   `Spectralon_ref-020.xlsx`, or **Constant reflectance (%)**), choose per-band or row-wise
   factors, and check the camera values the result will be bound to. While the dialog is open
   the pipeline keeps the newest 256 binned lines from before the reflectance step.
3. **Draw on RAW**: the belt map freezes (lines that weren't kept are dimmed). Drag a
   rectangle over the standard in the RAW zone, and **Esc** cancels. The summary shows white
   counts, % of full scale, spread, factor range and weak bands. Then choose **Apply & Save**,
   **Draw again** or **Cancel**.

**Apply & Save** uses the factors from the next line on and switches Reflectance on. It writes
`profiles/<camera>/reflectance/refl_b<bin>_g<gain>.npy` (a camera without model/serial:
`reflectance/<source>.npy`), and the JSON beside it holds the camera values (backend, bin, `exposure_ms`, `gain`, sensor temperature), area, reference and
quality report. The previous profile is moved to `reflectance/history/<name>_<time>.*`.
At every start a profile made at another exposure is scaled to the camera's exposure (counts
are linear in exposure). Another gain is flagged in the step's tooltip and the log, not scaled.

**Binding to the camera values:**

| value | rule |
|---|---|
| bin, gain | must match: one file per bin/gain in the profile; another bin/gain is refused (Reflectance off, reason in the tooltip) |
| exposure | scaled to the camera's exposure (counts are linear in it after dark) |
| sensor temperature | flagged when more than `corrections.reflectance_max_dtemp_c` (5 °C) from the calibration |
| age | flagged after `corrections.reflectance_max_age_h` (24 h): the lamp drifts |
| dark step | flagged when the factors were made with dark on and dark is off now |

A name-based `reflectance/<source>.npy` must record its bin/gain in its JSON when the camera is
identified, otherwise it is refused. The **Sensor…** dialog shows ✓/⚠ for both the dark and the
reflectance of the values you pick.

### White reference from a folder (`white_reference`)

The `white_reference` section reads the white standard like `source` reads the data. At start
the newest file of `path` (+ `pattern`) **recorded with the data's camera backend and bin** is
run through the data's own chain and turned into factors. They are cached in
`reflectance/from_white/<source>.npy` and rebuilt only when the white, the table, the settings
or the data's bands change. If no white matches, the hand-made `reflectance/<source>.npy` is
used and the step's tooltip says which whites were skipped and why.

| key | meaning |
|---|---|
| `path`, `pattern` | white recording(s): file or folder + glob; `null` = off |
| `table` | reference reflectance of the standard (`.xlsx`/`.csv`: nm, R) |
| `kind` | `band` (mean over the target rows) or `row` (row-wise, hsi window `avg_window`) |
| `rows`, `frames` | `"auto"` (found by brightness), `null` (all) or `[first, last+1]` |
| `slit_offset` | the white was saved as a slit crop: sensor row where the crop starts |

The white's own exposure/gain select its dark. A white recorded at another exposure than
the data is scaled to the data's exposure. Configured now: `reflectance/spectralon-20.h5`
with `Spectralon_ref-020.xlsx`, `slit_offset` 453. That is the GenICam SWIR white (bin 2), so
it serves a GenICam source and gives the same factors as `first.npy` (difference 0.0). The
live POA camera skips it and keeps its profile's `refl_b2_g1.0.npy`. Put a POA white recording into
`reflectance/` (or point `path` at a folder) to have POA factors made at start.

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
With a live camera the box is read-only and shows the camera's measured line rate.

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

The left side is one vertical splitter: **raw spectrogram + focus | belt map | bottom pane**
(backend selector, status line, provenance info).

* Upper handle, or the **Spectrogram** button: shrink or fully collapse the real-time raw
  spectrogram preview. While collapsed it is not rendered at all (saves GUI time at high rates).
* **Focus** button: the focus aid beside the preview (below); hidden = no score computed.
* Lower handle: drag down to give the belt map more height; the info panel shrinks to nothing,
  the backend row and status line stay visible.
* Vertical handle between the images and **Material Analytics**, or the **Analytics** button:
  change the panel width or hide it completely; the images get the freed width.
* The layout is remembered between runs (Windows registry,
  `HKCU\Software\NARIT-TSC\PushbroomSortingSimulation`); delete that key to get the default
  1:2 split back.

## Focusing the camera (focus aid)

The **FOCUS** card beside the spectrogram preview follows hsi's scan-window sharpness sidebar:
put a still, high-contrast target under the slit (e.g. black/white bars across the belt), turn
the lens focus ring slowly and stop where the number is highest. A score every 6 GUI refreshes
(hsi's 0.2 s at the default `display.gui_hz` 30; with the pipeline running, about 3 per second
were measured at 250 lines/s), each preview frame scored once, nothing computed while the card
or the preview is hidden. One update (edge contrast + hsi's score) took ~11 ms at 250 lines/s.

* **Big number**: edge contrast of the raw frame in % per pixel (`focus.edge_contrast`): the
  frame is averaged over the other axis, then the RMS step between neighbouring pixels of that
  profile is divided by its mean. Independent of exposure and gain; ~1 ms per 1024 × 1280 frame.
* **Bar and %**: score / running peak; the peak decays by 0.9995 per score as in hsi. **↺**
  forgets it (do this after changing the target, exposure or bin).
* **Trend**: the last 150 scores (~50 s at ~3 per second); the green dot is the best score in that window, so after turning
  past focus you can see how far to turn back.
* **hsi score**: hsi's `sharpness_score` formula (mean |Sobel|, `modules/sharpness/core.py`, a
  byte-identical copy) on the same raw frame, for comparison. hsi's scan window applies it to
  its dark/Smile-corrected, binned frame, so its numbers are not the same as this line's.
* **Slit / λ**: `Slit` (default) = imaging focus along the slit; `λ` = sharpness along the
  wavelength axis, for focusing the spectrograph on a line lamp.

Why the bar does not use hsi's score: the mean *absolute* gradient across an edge equals the
edge height however much it is blurred, and on raw frames the sensor noise (~500 counts sd here)
sets a high floor. On three recorded frames in `data/`, defocused by 1 px (Gaussian sigma, noise
added after the blur as in a real lens), hsi's score kept 93.5-95.1 % of its in-focus value,
edge contrast 71-96 % (on one frame slightly less responsive than hsi's up to 2 px, ahead from
3 px; at 5 px 36-73 % against 81-88 %). This is one simulation: the noise level is estimated
from the frames. On a bar target hsi's score stays flat up to 2 px of blur while edge contrast
falls to 53 % at 1 px; `tests/test_focus.py` pins this down.

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
covers sources, calibration resolution, model binding, confidence = sklearn `predict_proba`,
temperature scaling, lossless ordering, overrun counting
the bit-exact reproduction of the training cube, live backend switching and that the
processing code is still identical to hsi; `test_hsi_package.py` builds an HSI MLOps package and
checks reading without `hsi_lab_core`, the classes against a float64 reference, binding of all
training bands, dead pixels, refusals and that a package's pickle runs no code; `test_live_camera.py` runs the live camera source
against a fake camera backend (no hardware: configuration, calibration from the camera's
values, frame-id gaps and overruns, timestamps, stop/close); `test_themes.py` checks the themes;
`test_focus.py` the focus score and the focus card.
