"""Debug log files of every smile / keystone analysis, to check results and improve the system later.

One text file per analysis in ``modules/smile/logs/`` named after the local time it started,
``YYYYMMDD-HH-MM-SS.txt`` (e.g. ``20261119-14-22-05.txt``; ``-`` because Windows does not allow ``:``
in file names; a second analysis in the same second gets ``_2``, ``_3`` ...). A file holds:

* the environment (OS, machine, CPUs, Python / NumPy / OpenCV versions, thread);
* every input frame's statistics (shape, dtype, min / max / percentiles, share at full scale) and
  the parameters;
* the log messages of ``modules.smile`` during the analysis (DEBUG level, e.g. stage timings);
* the result: per line / per bar tables, fit and QC numbers, warnings, or the traceback of a
  failure;
* at the end, the same as one JSON object (``--- JSON ---``) for scripts.

Writing is off for plain library calls and tests. The smile widget and the command-line tools
turn it on (:func:`enable`); the environment variable ``HSI_SMILE_DEBUG_LOG`` (``1`` / ``0``)
overrides both, and ``HSI_SMILE_LOG_DIR`` moves the folder. Only the newest ``KEEP`` files are
kept. A failure to write a log never fails the analysis.

Threading: analyses may run in worker threads; each call writes its own file. While two analyses
overlap, the module log messages of both may appear in both files (each line names its thread).

Example:
    >>> from modules.smile import debug_log
    >>> debug_log.enable()                        # doctest: +SKIP
    >>> analyze_smile(frame)                      # doctest: +SKIP  (writes logs/20261119-14-22-05.txt)
"""
from __future__ import annotations

import functools
import json
import logging
import math
import os
import platform
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

__all__ = ["enable", "disable", "enabled", "log_dir", "logged_analysis", "KEEP"]

#: Newest log files kept in the folder (older ones are removed when a new one is written).
KEEP = 500
_DEFAULT_DIR = Path(__file__).resolve().parent / "logs"
_state = {"enabled": False, "dir": None}
_lock = threading.Lock()
_package_logger = logging.getLogger("modules.smile")


def enable(folder: Optional[str] = None) -> None:
    """Write a debug log for every following analysis (the app / CLIs call this)."""
    _state["enabled"] = True
    if folder is not None:
        _state["dir"] = Path(folder)


def disable() -> None:
    """Stop writing debug logs."""
    _state["enabled"] = False


def enabled() -> bool:
    """True when analyses write a log: ``HSI_SMILE_DEBUG_LOG`` if set, else :func:`enable` state."""
    env = os.environ.get("HSI_SMILE_DEBUG_LOG")
    if env is not None and env.strip() != "":
        return env.strip().lower() not in ("0", "false", "no", "off")
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("HSI_SMILE_LOG_DIR"):
        return False                     # tests create widgets (which enable logging): no files in the repo
    return bool(_state["enabled"])


def log_dir() -> Path:
    """The folder of the log files (``HSI_SMILE_LOG_DIR``, :func:`enable` folder, or ``modules/smile/logs``)."""
    env = os.environ.get("HSI_SMILE_LOG_DIR")
    return Path(env) if env else (_state["dir"] or _DEFAULT_DIR)


# ----------------------------------------------------------------------------------------- helpers
def _new_path(start: datetime) -> Path:
    folder = log_dir()
    folder.mkdir(parents=True, exist_ok=True)
    gi = folder / ".gitignore"
    if not gi.exists():
        try:
            gi.write_text("# debug logs of the smile / keystone analyses: not part of the code\n*\n!.gitignore\n",
                          encoding="utf-8")
        except OSError:
            pass
    stem = start.strftime("%Y%m%d-%H-%M-%S")
    with _lock:
        path, n = folder / f"{stem}.txt", 1
        while path.exists():
            n += 1
            path = folder / f"{stem}_{n}.txt"
        path.touch()
    return path


def _prune() -> None:
    try:
        files = sorted(log_dir().glob("*.txt"), key=lambda p: p.stat().st_mtime)
        for p in files[:-KEEP] if len(files) > KEEP else []:
            p.unlink(missing_ok=True)
    except OSError:
        pass


def _environment() -> dict:
    try:
        import cv2
        cv = cv2.__version__
    except Exception:            # noqa: BLE001
        cv = None
    try:
        import scipy
        sp = scipy.__version__
    except Exception:            # noqa: BLE001
        sp = None
    return dict(os=platform.platform(), machine=platform.machine(), cpus=os.cpu_count(),
                python=sys.version.split()[0], numpy=np.__version__, scipy=sp, opencv=cv,
                thread=threading.current_thread().name, pid=os.getpid())


def _frame_stats(a) -> dict:
    a = np.asarray(a)
    out = dict(shape=list(a.shape), dtype=str(a.dtype))
    if a.size == 0 or not np.issubdtype(a.dtype, np.number):
        return out
    f = a[::max(1, a.shape[0] // 512)] if a.ndim >= 1 else a      # subsample rows: fast on big frames
    fin = np.isfinite(f) if np.issubdtype(f.dtype, np.floating) else np.ones(f.shape, bool)
    v = f[fin].astype(np.float64)
    if v.size == 0:
        return out
    p = np.percentile(v, (1, 50, 99, 99.9))
    out.update(min=float(v.min()), max=float(v.max()), p1=float(p[0]), median=float(p[1]), p99=float(p[2]),
               p99_9=float(p[3]), mean=float(v.mean()), nonfinite=int((~fin).sum()))
    vmax = float(v.max())
    if vmax > 0:
        out["share_at_max"] = float(np.mean(v >= vmax))
    return out


def _jsonable(v: Any, depth: int = 0) -> Any:
    if depth > 12:
        return str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, depth + 1) for x in v]
    if isinstance(v, np.ndarray):
        return _frame_stats(v) if v.ndim >= 2 else _jsonable(v.tolist(), depth + 1)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    return repr(v)[:300]


def _compact(v):
    """JSON-able value with floats to 6 significant digits (for the readable part)."""
    v = _jsonable(v)
    if isinstance(v, float):
        return float("%.6g" % v)
    if isinstance(v, dict):
        return {k: _compact(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_compact(x) for x in v]
    return v


def _lines(d: dict) -> list:
    """One readable line per top-level key: key: compact JSON."""
    if not d:
        return ["(none)"]
    return ["%s: %s" % (k, json.dumps(_compact(v), ensure_ascii=False, separators=(", ", ": ")))
            for k, v in d.items()]


def _fmt(v, nd=4) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "-" if not math.isfinite(f) else f"{f:.{nd}f}"


# ----------------------------------------------------------------------------------------- summaries
def _summary_smile(r) -> tuple:
    """(text, json) of a SmileAnalysis."""
    lines = [f"y_ref {r.y_ref}, interp {r.interp}, sat_level {_fmt(r.sat_level, 1)}, noise {_fmt(r.noise, 3)}",
             f"surface deg_y {r.surface.deg_y} deg_x {r.surface.deg_x}, y_rng {r.surface.y_rng}, x_rng {r.surface.x_rng}",
             f"straightness RMS before {_fmt(r.rms_before)} px, after {_fmt(r.rms_after)} px, "
             f"leave-one-line-out {_fmt(r.cv_rms)} px, fit RMS {_fmt(r.fit_info.get('rms'))} px "
             f"({r.fit_info.get('n_used')} / {r.fit_info.get('n')} points)",
             f"margin (left, right) {tuple(r.margin)}",
             "timings: " + ", ".join(f"{k} {v:.3f} s" for k, v in (r.timings or {}).items()),
             "",
             "lines  column    fwhm     height  saturated  used  rows_traced  reference  resid_rms"]
    used = {ln.column: i for i, ln in enumerate(r.used_lines)}
    table = []
    for ln in r.all_lines:
        c = np.asarray(r.traces_orig.get(ln.column, []), float)
        n = int(np.isfinite(c).sum()) if c.size else 0
        k = used.get(ln.column)
        res = None
        if k is not None:
            sel = np.asarray(r.meas_id) == k
            res = float(np.sqrt(np.mean(np.square(np.asarray(r.residuals)[sel])))) if sel.any() else None
        ref = r.refs.get(ln.column)
        lines.append(f"       {ln.column:7.1f} {ln.fwhm:7.2f} {ln.height:10.1f}  {str(ln.saturated):9s}  "
                     f"{'yes' if k is not None else 'no':4s}  {n:11d}  {_fmt(ref, 3):>9s}  {_fmt(res, 4):>9s}")
        table.append(dict(column=ln.column, fwhm=ln.fwhm, height=ln.height, saturated=ln.saturated,
                          used=k is not None, rows_traced=n, reference=ref, residual_rms=res))
    js = dict(y_ref=r.y_ref, interp=r.interp, sat_level=r.sat_level, noise=r.noise,
              surface=dict(deg_y=r.surface.deg_y, deg_x=r.surface.deg_x, coef=list(map(float, r.surface.coef)),
                           y_rng=list(r.surface.y_rng), x_rng=list(r.surface.x_rng)),
              fit_info={k: v for k, v in r.fit_info.items() if k != "coef"}, rms_before=r.rms_before,
              rms_after=r.rms_after, cv_rms=r.cv_rms, margin=list(r.margin), timings=r.timings, lines=table)
    return "\n".join(lines), js


def _summary_keystone(k) -> tuple:
    """(text, json) of a KeystoneAnalysis."""
    from .keystone import keystone_report
    d = k.to_json()
    q = d.get("qc") or {}
    lines = [keystone_report(d), "", "bars  y_seed  coverage  halfwin  points  fit_sigma  weight  retrace_rms  retrace_max"]
    per = q.get("retrace_target_per_bar") or []
    sig = q.get("fit_target_per_bar_px") or []
    wts = q.get("bar_weight") or []
    for i, b in enumerate(k.target_lines):
        rt = per[i] if i < len(per) else {}
        lines.append(f"      {b.y_seed:7.1f}  {b.coverage:8.2f}  {b.halfwin:7d}  {b.x.size:6d}  "
                     f"{_fmt(sig[i] if i < len(sig) else None):>9s}  {_fmt(wts[i] if i < len(wts) else None, 3):>6s}  "
                     f"{_fmt(rt.get('rms_px')):>11s}  {_fmt(rt.get('max_px')):>11s}")
    lp = q.get("retrace_lamp_per_line") or []
    if lp:
        lines += ["", "lamp lines (re-traced on the corrected frame)  X  rms  max  n"]
        lines += [f"      {_fmt(p.get('X'), 1):>8s}  {_fmt(p.get('rms_px')):>8s}  {_fmt(p.get('max_px')):>8s}  {p.get('n')}"
                  for p in lp]
    return "\n".join(lines), d


def _summary(result) -> tuple:
    try:
        if hasattr(result, "used_lines") and hasattr(result, "surface"):
            return _summary_smile(result)
        if hasattr(result, "target_lines") and hasattr(result, "model"):
            return _summary_keystone(result)
    except Exception:            # noqa: BLE001 - a log must never fail the analysis
        return "summary failed:\n" + traceback.format_exc(), None
    return repr(result)[:2000], None


# ----------------------------------------------------------------------------------------- decorator
class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []
        self.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(threadName)s %(name)s %(levelname)s: %(message)s",
                                            "%H:%M:%S"))

    def emit(self, record):
        try:
            self.lines.append(self.format(record))
        except Exception:        # noqa: BLE001
            pass


def logged_analysis(kind: str) -> Callable:
    """Decorator: when :func:`enabled`, write a debug log file for each call of an analysis."""
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            start = datetime.now()
            t0 = time.perf_counter()
            handler = _Collector()
            old_level = _package_logger.level
            _package_logger.addHandler(handler)
            if _package_logger.getEffectiveLevel() > logging.DEBUG:
                _package_logger.setLevel(logging.DEBUG)
            result, error = None, None
            try:
                result = fn(*args, **kwargs)
                return result
            except BaseException as exc:          # noqa: BLE001 - logged, then raised again
                error = exc
                raise
            finally:
                _package_logger.removeHandler(handler)
                _package_logger.setLevel(old_level)
                try:
                    _write(kind, fn, args, kwargs, start, time.perf_counter() - t0, handler.lines, result, error)
                except Exception:                 # noqa: BLE001 - never fail the analysis because of a log
                    logging.getLogger(__name__).warning("Could not write the smile debug log", exc_info=True)
        return wrapper
    return deco


def _write(kind, fn, args, kwargs, start, elapsed, messages, result, error) -> None:
    path = _new_path(start)
    inputs = {}
    params = {}
    try:
        import inspect
        names = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        names = []
    for i, a in enumerate(args):
        key = names[i] if i < len(names) else f"arg{i}"
        if isinstance(a, np.ndarray):
            inputs[key] = _frame_stats(a)
        elif hasattr(a, "frame_orig"):                          # a SmileAnalysis passed in
            inputs[key] = dict(smile_analysis=True, frame=_frame_stats(a.frame_orig),
                               lines_used=len(a.used_lines), cv_rms=a.cv_rms)
        elif hasattr(a, "deg_y") and hasattr(a, "coef"):        # a stored Surface2D
            inputs[key] = dict(stored_smile_surface=True, deg_y=a.deg_y, deg_x=a.deg_x,
                               coef=list(map(float, a.coef)), y_rng=list(a.y_rng), x_rng=list(a.x_rng))
        else:
            params[key] = _jsonable(a)
    for k, v in kwargs.items():
        (inputs if isinstance(v, np.ndarray) else params)[k] = _frame_stats(v) if isinstance(v, np.ndarray) else _jsonable(v)
    env = _environment()
    status = "FAILED" if error is not None else "OK"
    text, js = (None, None) if error is not None else _summary(result)
    out = [f"{kind} analysis - {status}",
           f"started {start:%Y-%m-%d %H:%M:%S}, {elapsed:.3f} s, function {fn.__module__}.{fn.__name__}",
           "", "=== environment ==="] + _lines(env) + [
           "", "=== inputs ==="] + _lines(inputs) + [
           "", "=== parameters ==="] + _lines(params) + [
           "", "=== messages ==="] + (messages or ["(none)"])
    if error is not None:
        out += ["", "=== error ===", "".join(traceback.format_exception(type(error), error, error.__traceback__))]
    else:
        out += ["", "=== result ===", text or ""]
    out += ["", "--- JSON ---", json.dumps(_jsonable(dict(kind=kind, status=status, started=start.isoformat(),
                                                           seconds=elapsed, environment=env, inputs=inputs,
                                                           parameters=params, messages=messages,
                                                           error=None if error is None else repr(error),
                                                           result=js)), indent=1)]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    _prune()
