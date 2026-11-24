# -*- coding: utf-8 -*-
"""Visual report of a model trained by ``tools/train_sgd_coco.py``.

``write_report`` writes into one folder:

    overview.png      confusion matrix (held-out), precision / recall / F1 per class,
                      labelled pixels per cube and class, held-out accuracy per cube
    spectra.png       mean spectrum +- 1 std of each class, the bands SelectKBest kept, F-score per band
    weights.png       model weight of every kept band for every class (standardised bands)
    calibration.png   reliability diagram before / after the temperature, confidence of right and wrong pixels
    projection.png    held-out pixels on the 2 LDA axes of the kept bands, misclassified pixels marked
    map_<cube>.png    per cube: image, labels, predicted class, confidence, errors on the labelled pixels
    report.txt        the numbers of the figures
    index.html        every figure on one page (open it in a browser)

Only matplotlib's object API is used (no pyplot), so it runs headless and does not touch a Qt app.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Categorical slots in fixed order (validated palette), then extras only past 8 classes.
CLASS_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948',
                '#7f7f7f', '#17becf', '#8c564b', '#bcbd22']
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e4e3df'
WRONG = '#d03b3b'
RIGHT = '#a9a8a2'
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#f1f5fb', '#9cc0ec', '#2a78d6', '#0d3d7a'])
DIV = LinearSegmentedColormap.from_list('div_or_bl', ['#b8461a', '#eb6834', '#e9e8e4', '#2a78d6', '#164f96'])
MAX_SCATTER_PER_CLASS = 3000
LINES_PER_CHUNK = 64


@dataclass
class ReportInputs:
    data: object                    # train_sgd_coco.TrainingSet (needs ``position``)
    idx_train: np.ndarray
    idx_test: np.ndarray
    selector: object                # fitted SelectKBest
    scaler: object                  # fitted StandardScaler
    model: object                   # fitted SGDClassifier
    temperature: float
    artifact: dict                  # W_fused, b_fused, selected_bands, classes


def _style(ax, grid_axis='y'):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _figure(w, h):
    return Figure(figsize=(w, h), dpi=110, facecolor=SURFACE, layout='constrained')


def _save(fig, path: Path) -> Path:
    fig.savefig(path, facecolor=SURFACE)
    return path


def _ovr_probabilities(scores, temperature=1.0):
    p = 1.0 / (1.0 + np.exp(-np.clip(scores / temperature, -80.0, 80.0)))
    return p / p.sum(axis=1, keepdims=True)


class _Ctx:
    """Everything the figures share, computed once."""

    def __init__(self, r: ReportInputs):
        d = r.data
        self.r = r
        self.classes = np.asarray(r.model.classes_)
        self.names = [str(d.category_names.get(int(c), c)) for c in self.classes]
        self.colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(self.classes))]
        self.sel = np.asarray(r.artifact['selected_bands'])
        self.wl = np.asarray(d.wavelengths, float)

        def scaled(idx):
            return r.scaler.transform(r.selector.transform(d.X[idx]))
        self.Xs_train, self.Xs_test = scaled(r.idx_train), scaled(r.idx_test)
        self.y_train, self.y_test = d.y[r.idx_train], d.y[r.idx_test]
        self.pred_train = r.model.predict(self.Xs_train)
        self.pred_test = r.model.predict(self.Xs_test)
        self.scores_test = r.model.decision_function(self.Xs_test)
        self.conf_raw = _ovr_probabilities(self.scores_test).max(axis=1)
        self.conf_t = _ovr_probabilities(self.scores_test, r.temperature).max(axis=1)
        self.correct = self.pred_test == self.y_test
        self.cm = confusion_matrix(self.y_test, self.pred_test, labels=self.classes)
        self.prf = precision_recall_fscore_support(self.y_test, self.pred_test, labels=self.classes,
                                                   zero_division=0)
        self.cube_names = [Path(h).name for h, _ in d.cubes]


# ---------------------------------------------------------------- overview

def _confusion(ax, c: _Ctx):
    cm = c.cm.astype(float)
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    ax.imshow(pct, cmap=SEQ, vmin=0, vmax=100)
    n = len(c.classes)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{pct[i, j]:.1f}%\n{int(cm[i, j]):,}", ha='center', va='center', fontsize=8,
                    color='white' if pct[i, j] > 55 else INK)
    ax.set_xticks(range(n), c.names, rotation=30, ha='right')
    ax.set_yticks(range(n), c.names)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true (labelled)')
    acc = c.correct.mean()
    ax.set_title(f"Confusion matrix, held-out pixels (row %)\naccuracy {acc:.4f} of {c.y_test.size:,} pixels",
                 fontsize=10, loc='left')
    _style(ax, grid_axis=None)
    for s in ax.spines.values():
        s.set_visible(False)


def _metrics(ax, c: _Ctx):
    precision, recall, f1, support = c.prf
    n = len(c.classes)
    x = np.arange(n)
    w = 0.26
    shades = ['#b9d2f2', '#6a9fe2', '#1d5fae']
    for k, (vals, label) in enumerate(((precision, 'precision'), (recall, 'recall'), (f1, 'F1'))):
        bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=shades[k], label=label)
        if label == 'F1':
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha='center', va='bottom',
                        fontsize=7.5, color=INK)
    low = float(np.min([precision, recall, f1])) if n else 0.0
    ax.set_ylim(max(0.0, low - 0.1 * (1 - low) - 0.02), 1.0 + 0.04 * (1 - max(0.0, low - 0.1)))
    ax.set_xticks(x, [f"{nm}\n(n={int(s):,})" for nm, s in zip(c.names, support)])
    train_acc = np.mean(c.pred_train == c.y_train)
    ax.set_title(f"Per-class metrics, held-out\ntrain accuracy {train_acc:.4f} vs held-out "
                 f"{c.correct.mean():.4f}", fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    _style(ax)


def _pixels_per_cube(ax, c: _Ctx):
    d = c.r.data
    left = np.zeros(len(c.cube_names))
    for k, (cls, nm, col) in enumerate(zip(c.classes, c.names, c.colors)):
        counts = np.array([np.sum((d.source == i) & (d.y == cls)) for i in range(len(c.cube_names))], float)
        ax.barh(range(len(c.cube_names)), counts, left=left, color=col, label=nm,
                edgecolor=SURFACE, linewidth=2, height=0.6)
        left += counts
    for i, total in enumerate(left):
        ax.text(total, i, f"  {int(total):,}", va='center', fontsize=8, color=INK_2)
    ax.set_yticks(range(len(c.cube_names)), c.cube_names)
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.18 if left.size else 1)
    ax.set_xlabel('labelled pixels')
    ax.set_title('Labelled pixels per cube and class', fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    _style(ax, grid_axis='x')


def _accuracy_per_cube(ax, c: _Ctx):
    src = c.r.data.source[c.r.idx_test]
    accs, ns = [], []
    for i in range(len(c.cube_names)):
        m = src == i
        accs.append(np.mean(c.correct[m]) if m.any() else np.nan)
        ns.append(int(m.sum()))
    accs = np.array(accs)
    ax.barh(range(len(accs)), np.nan_to_num(accs), color='#2a78d6', height=0.5)
    for i, (a, n) in enumerate(zip(accs, ns)):
        ax.text(np.nan_to_num(a), i, f"  {a:.4f}  ({n:,} px)", va='center', fontsize=8, color=INK)
    ax.set_yticks(range(len(accs)), c.cube_names)
    ax.invert_yaxis()
    low = np.nanmin(accs) if np.isfinite(accs).any() else 0.0
    start = max(0.0, np.floor((low - 0.02) * 50) / 50)
    ax.set_xlim(start, 1.0 + (1.0 - start) * 0.45)
    ax.set_xticks([t for t in ax.get_xticks() if start <= t <= 1.0 + 1e-9])
    ax.set_xlabel('held-out accuracy')
    ax.set_title('Held-out accuracy per cube', fontsize=10, loc='left')
    _style(ax, grid_axis='x')


def fig_overview(c: _Ctx, path: Path) -> Path:
    fig = _figure(13, 9)
    axes = fig.subplots(2, 2)
    _confusion(axes[0, 0], c)
    _metrics(axes[0, 1], c)
    _pixels_per_cube(axes[1, 0], c)
    _accuracy_per_cube(axes[1, 1], c)
    return _save(fig, path)


# ---------------------------------------------------------------- spectra

def fig_spectra(c: _Ctx, path: Path) -> Path:
    d = c.r.data
    fig = _figure(13, 8)
    ax, ax_f = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1.4])
    step = np.median(np.diff(c.wl)) if c.wl.size > 1 else 1.0
    for b in c.sel:
        ax.axvspan(c.wl[b] - step / 2, c.wl[b] + step / 2, color='#ecebe6', lw=0)
    X_train = d.X[c.r.idx_train]
    for cls, nm, col in zip(c.classes, c.names, c.colors):
        Xc = X_train[c.y_train == cls]
        if not len(Xc):
            continue
        mu, sd = Xc.mean(axis=0), Xc.std(axis=0)
        ax.fill_between(c.wl, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.plot(c.wl, mu, color=col, lw=2, label=f"{nm} (n={len(Xc):,})")
        ax.annotate(nm, (c.wl[-1], mu[-1]), xytext=(4, 0), textcoords='offset points', va='center',
                    fontsize=8, color=INK)
    ax.set_ylabel(d.input_units)
    ax.set_title(f"Mean spectrum per class (training pixels, band = +-1 std); "
                 f"grey columns = the {len(c.sel)} bands kept by SelectKBest", fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8, loc='best')
    _style(ax)

    f_scores = np.nan_to_num(np.asarray(c.r.selector.scores_, float))
    keep = np.zeros(c.wl.size, bool)
    keep[c.sel] = True
    ax_f.bar(c.wl[~keep], f_scores[~keep], width=step * 0.85, color='#cfcec8', label='not used')
    ax_f.bar(c.wl[keep], f_scores[keep], width=step * 0.85, color='#2a78d6', label='kept')
    if np.all(f_scores > 0) and f_scores.max() / max(f_scores.min(), 1e-12) > 100:
        ax_f.set_yscale('log')
    ax_f.set_ylabel('ANOVA F')
    ax_f.set_xlabel('wavelength (nm)')
    ax_f.set_title('Band separability (F-score, training pixels)', fontsize=10, loc='left')
    ax_f.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')
    _style(ax_f)
    return _save(fig, path)


# ---------------------------------------------------------------- weights

def fig_weights(c: _Ctx, path: Path) -> Path:
    coef = np.asarray(c.r.model.coef_, float)                        # (classes, kept bands), standardised
    lim = float(np.abs(coef).max()) or 1.0
    fig = _figure(max(8, 0.32 * len(c.sel) + 3), 1.0 + 0.55 * len(c.classes) + 1.4)
    ax = fig.subplots()
    im = ax.imshow(coef, cmap=DIV, vmin=-lim, vmax=lim, aspect='auto')
    ax.set_yticks(range(len(c.classes)), c.names)
    ax.set_xticks(range(len(c.sel)), [f"{c.wl[b]:.0f}" for b in c.sel], rotation=90)
    ax.set_xlabel('kept band (nm)')
    ax.set_title('Model weight per kept band and class (bands standardised: blue pushes towards the class, '
                 'orange away)', fontsize=10, loc='left')
    gaps = np.flatnonzero(np.diff(c.sel) > 1)              # kept bands that are not neighbours
    for g in gaps:
        ax.axvline(g + 0.5, color=SURFACE, lw=3)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    cb.outline.set_visible(False)
    _style(ax, grid_axis=None)
    return _save(fig, path)


# ---------------------------------------------------------------- calibration

def _reliability(conf, correct, bins=10):
    edges = np.linspace(1.0 / bins * 0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    centers, accs, counts = [], [], []
    for b in range(bins):
        m = idx == b
        if m.any():
            centers.append(conf[m].mean())
            accs.append(correct[m].mean())
            counts.append(int(m.sum()))
    centers, accs, counts = map(np.asarray, (centers, accs, counts))
    ece = float(np.sum(counts / max(conf.size, 1) * np.abs(accs - centers))) if counts.size else 0.0
    return centers, accs, counts, ece


def fig_calibration(c: _Ctx, path: Path) -> Path:
    fig = _figure(13, 5.2)
    ax, ax_h = fig.subplots(1, 2)
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls='--', label='perfect')
    rows = []
    for conf, label, col in ((c.conf_raw, 'raw (T = 1)', '#cfcec8'),
                             (c.conf_t, f'temperature T = {c.r.temperature:.3f}', '#2a78d6')):
        x, a, n, ece = _reliability(conf, c.correct)
        ax.plot(x, a, marker='o', ms=6, lw=2, color=col, label=f"{label}, ECE {ece:.4f}")
        rows.append((label, ece))
    for xi, ai, ni in zip(x, a, n):                        # pixels per bin of the temperature curve
        ax.annotate(f"{ni:,}", (xi, ai), xytext=(6, -10), textcoords='offset points', fontsize=7, color=INK_2)
    lo = min(0.95, float(min(c.conf_raw.min(), c.conf_t.min())))
    lo = max(0.0, np.floor(lo * 10) / 10)
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.005)
    ax.set_xlabel('confidence of the winning class')
    ax.set_ylabel('fraction right')
    ax.set_title('Reliability, held-out pixels (on the diagonal = honest confidence;\n'
                 'numbers = pixels per bin, few pixels = noisy point)', fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _style(ax, grid_axis='both')

    bins = np.linspace(lo, 1.0, 41)
    ax_h.hist(c.conf_t[c.correct], bins=bins, color=RIGHT, label=f"right ({int(c.correct.sum()):,})")
    ax_h.hist(c.conf_t[~c.correct], bins=bins, color=WRONG, alpha=0.9,
              label=f"wrong ({int((~c.correct).sum()):,})")
    ax_h.set_yscale('log')
    ax_h.set_xlabel(f'confidence (T = {c.r.temperature:.3f})')
    ax_h.set_ylabel('pixels (log)')
    ax_h.set_title('Confidence of right and wrong held-out pixels', fontsize=10, loc='left')
    ax_h.legend(frameon=False, fontsize=8, loc='upper left')
    _style(ax_h)
    return _save(fig, path)


# ---------------------------------------------------------------- projection

def fig_projection(c: _Ctx, path: Path) -> Path | None:
    if len(c.classes) < 3:
        return None
    lda = LinearDiscriminantAnalysis(n_components=2).fit(c.Xs_train, c.y_train)
    rng = np.random.default_rng(0)
    pick = []
    for cls in c.classes:
        idx = np.flatnonzero(c.y_test == cls)
        pick.append(rng.choice(idx, min(idx.size, MAX_SCATTER_PER_CLASS), replace=False) if idx.size else idx)
    pick = np.concatenate(pick)
    wrong = np.flatnonzero(~c.correct)
    Z = lda.transform(c.Xs_test[pick])
    Zw = lda.transform(c.Xs_test[wrong]) if wrong.size else np.zeros((0, 2))
    fig = _figure(9, 7.5)
    ax = fig.subplots()
    for cls, nm, col in zip(c.classes, c.names, c.colors):
        m = c.y_test[pick] == cls
        ax.scatter(Z[m, 0], Z[m, 1], s=6, color=col, alpha=0.45, lw=0, label=nm)
    if wrong.size:
        ax.scatter(Zw[:, 0], Zw[:, 1], s=14, marker='x', color=INK, lw=0.8, alpha=0.7,
                   label=f"misclassified ({wrong.size:,})")
    for cls, nm in zip(c.classes, c.names):
        m = c.y_test[pick] == cls
        if m.any():
            cx, cy = np.median(Z[m], axis=0)
            ax.annotate(nm, (cx, cy), fontsize=9, color=INK, ha='center', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc=SURFACE, ec='none', alpha=0.8))
    ax.set_xlabel('LDA axis 1')
    ax.set_ylabel('LDA axis 2')
    ax.set_title(f"Held-out pixels on the 2 most separating axes (LDA of the kept bands)\n"
                 f"up to {MAX_SCATTER_PER_CLASS:,} per class + every misclassified pixel (x)", fontsize=10, loc='left')
    leg = ax.legend(frameon=False, fontsize=8, loc='best', markerscale=2)
    for h in leg.legend_handles:
        h.set_alpha(1)
    _style(ax, grid_axis='both')
    return _save(fig, path)


# ---------------------------------------------------------------- maps per cube

def _predict_cube(cube, art, temperature):
    """(class index map, confidence map) of the whole cube with the fused weights."""
    lines, samples, _ = cube.shape
    sel = np.asarray(art['selected_bands'])
    W = np.asarray(art['W_fused'], np.float32)
    b = np.asarray(art['b_fused'], np.float32)
    cls_map = np.zeros((lines, samples), np.int16)
    conf_map = np.zeros((lines, samples), np.float32)
    for l0 in range(0, lines, LINES_PER_CHUNK):
        x = np.asarray(cube[l0:l0 + LINES_PER_CHUNK], np.float32)[..., sel]
        s = x.reshape(-1, sel.size) @ W + b
        p = _ovr_probabilities(s, temperature)
        n = x.shape[0]
        cls_map[l0:l0 + n] = p.argmax(axis=1).reshape(n, samples)
        conf_map[l0:l0 + n] = p.max(axis=1).reshape(n, samples)
    return cls_map, conf_map


def _gray(cube):
    img = np.zeros(cube.shape[:2], np.float32)
    for l0 in range(0, cube.shape[0], LINES_PER_CHUNK):
        img[l0:l0 + LINES_PER_CHUNK] = np.asarray(cube[l0:l0 + LINES_PER_CHUNK], np.float32).mean(axis=2)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1)


def fig_cube_maps(c: _Ctx, i: int, path: Path) -> tuple[Path, float]:
    import spectral as spy
    d, r = c.r.data, c.r
    hdr = d.cubes[i][0]
    cube = spy.open_image(str(hdr)).open_memmap()
    lines, samples, _ = cube.shape
    gray = _gray(cube)
    cls_map, conf_map = _predict_cube(cube, r.artifact, r.temperature)

    of_cube = np.flatnonzero(d.source == i)
    pos = d.position[of_cube]
    class_index = {int(cls): k for k, cls in enumerate(c.classes)}
    label_map = np.full(lines * samples, -1, np.int16)
    label_map[pos] = [class_index.get(int(v), -1) for v in d.y[of_cube]]
    label_map = label_map.reshape(lines, samples)
    is_test = np.zeros(d.y.size, bool)
    is_test[r.idx_test] = True
    status = np.full(lines * samples, np.nan, np.float32)          # 0 right, 2 wrong
    pred_flat = cls_map.reshape(-1)[pos]
    truth = label_map.reshape(-1)[pos]
    wrong = pred_flat != truth
    status[pos] = np.where(wrong, 2, 0)
    n_wrong_test = int(np.sum(wrong & is_test[of_cube]))
    status = status.reshape(lines, samples)
    acc_all = float(np.mean(~wrong)) if pos.size else float('nan')

    cmap_cls = ListedColormap(c.colors)
    fig = _figure(20, 7.2 * min(2.0, max(0.6, lines / max(samples, 1) / 1.6)))
    axes = fig.subplots(1, 5)
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('cube (mean of all bands)', fontsize=9, loc='left')
    axes[1].imshow(gray, cmap='gray')
    axes[1].imshow(np.ma.masked_less(label_map, 0), cmap=cmap_cls, vmin=-0.5, vmax=len(c.classes) - 0.5,
                   interpolation='nearest', alpha=0.85)
    axes[1].set_title(f'labels ({pos.size:,} px)', fontsize=9, loc='left')
    axes[2].imshow(cls_map, cmap=cmap_cls, vmin=-0.5, vmax=len(c.classes) - 0.5, interpolation='nearest')
    frac = np.bincount(cls_map.reshape(-1), minlength=len(c.classes)) / cls_map.size * 100
    axes[2].set_title('predicted class (whole cube)', fontsize=9, loc='left')
    im = axes[3].imshow(conf_map, cmap=SEQ, vmin=max(0.0, float(np.percentile(conf_map, 1))), vmax=1.0)
    axes[3].set_title(f'confidence (T = {r.temperature:.3f}), median {np.median(conf_map):.3f}',
                      fontsize=9, loc='left')
    cb = fig.colorbar(im, ax=axes[3], shrink=0.6, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)
    axes[4].imshow(gray * 0.45, cmap='gray', vmin=0, vmax=1)
    axes[4].imshow(status, cmap=ListedColormap(['#d9d8d2', '#d9d8d2', '#ff4d4d']), vmin=-0.5, vmax=2.5,
                   interpolation='nearest')
    n_wrong = int(wrong.sum())
    axes[4].set_title(f'errors on labelled pixels: {n_wrong:,} wrong ({acc_all:.4f} right)\n'
                      f'{n_wrong_test:,} of them held-out', fontsize=9, loc='left')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    class_handles = [Patch(color=col, label=f"{nm} ({f:.1f}% of cube)")
                     for col, nm, f in zip(c.colors, c.names, frac)]
    status_handles = [Patch(color='#d9d8d2', label='labelled, right'), Patch(color='#ff4d4d', label='labelled, wrong')]
    fig.legend(handles=class_handles + status_handles, loc='outside lower center', ncol=6, frameon=False,
               fontsize=8)
    fig.suptitle(f"{Path(hdr).name}  -  {lines} lines x {samples} samples", fontsize=11, color=INK,
                 x=0.01, ha='left')
    return _save(fig, path), acc_all


# ---------------------------------------------------------------- text + html

def _text_report(c: _Ctx, cube_acc) -> str:
    d, r = c.r.data, c.r
    out = [f"Training cubes: {len(d.cubes)}  pixels: {d.y.size:,}  fit: {r.idx_train.size:,}  "
           f"held-out: {r.idx_test.size:,}  units: {d.input_units}",
           f"Kept bands ({len(c.sel)}): " + ", ".join(f"{c.wl[b]:.1f}" for b in c.sel) + " nm",
           f"Train accuracy {np.mean(c.pred_train == c.y_train):.4f}   held-out accuracy {c.correct.mean():.4f}",
           f"Temperature {r.temperature:.4f}   mean confidence {c.conf_raw.mean():.4f} -> {c.conf_t.mean():.4f}",
           "", "Classification report (held-out):",
           classification_report(c.y_test, c.pred_test, labels=c.classes, target_names=c.names, digits=4,
                                 zero_division=0),
           "Confusion matrix (rows true, columns predicted):",
           " " * 14 + "".join(f"{n[:10]:>11}" for n in c.names)]
    for nm, row in zip(c.names, c.cm):
        out.append(f"{nm[:13]:<14}" + "".join(f"{v:>11,}" for v in row))
    out += ["", f"{'cube':<42}{'held-out acc':>14}{'all labelled acc':>18}"]
    src = d.source[r.idx_test]
    for i, name in enumerate(c.cube_names):
        m = src == i
        held = np.mean(c.correct[m]) if m.any() else float('nan')
        out.append(f"{name:<42}{held:>14.4f}{cube_acc.get(i, float('nan')):>18.4f}")
    return "\n".join(out) + "\n"


def _html(title, text, figures) -> str:
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<style>body{font-family:system-ui,sans-serif;background:#fcfcfb;color:#0b0b0b;margin:0 auto;"
             "max-width:1500px;padding:16px}img{max-width:100%;border:1px solid #e4e3df;border-radius:6px}"
             "pre{background:#f3f2ee;padding:12px;border-radius:6px;overflow-x:auto;font-size:13px}"
             "h2{font-size:16px;margin:28px 0 8px;color:#52514e}</style></head><body>",
             f"<h1>{html.escape(title)}</h1><pre>{html.escape(text)}</pre>"]
    for heading, p in figures:
        parts.append(f"<h2>{html.escape(heading)}</h2><img src='{html.escape(p.name)}' alt='{html.escape(heading)}'>")
    parts.append("</body></html>")
    return "\n".join(parts)


def write_report(inputs: ReportInputs, out_dir, title: str = 'SGD training report') -> Path:
    """Write every figure, ``report.txt`` and ``index.html`` into ``out_dir``; returns ``index.html``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = _Ctx(inputs)
    figures = [('Overview', fig_overview(c, out / 'overview.png')),
               ('Spectra and band selection', fig_spectra(c, out / 'spectra.png')),
               ('Model weights', fig_weights(c, out / 'weights.png')),
               ('Confidence calibration', fig_calibration(c, out / 'calibration.png'))]
    proj = fig_projection(c, out / 'projection.png')
    if proj is not None:
        figures.append(('Class separation (LDA)', proj))
    cube_acc = {}
    for i, name in enumerate(c.cube_names):
        stem = name[:-len('.hdr')] if name.lower().endswith('.hdr') else name
        p, cube_acc[i] = fig_cube_maps(c, i, out / f"map_{stem.replace('+', '_')}.png")
        figures.append((f"Maps: {name}", p))
    text = _text_report(c, cube_acc)
    (out / 'report.txt').write_text(text, encoding='utf-8')
    index = out / 'index.html'
    index.write_text(_html(title, text, figures), encoding='utf-8')
    return index
