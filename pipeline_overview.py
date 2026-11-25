"""What each processing step does now, as blocks for the Setup tab's pipeline view.

``describe(pipeline, snapshot)`` returns ``(headline, blocks)``: one :class:`Block` per step from
the sensor to the model, each with a state (``ok`` / ``warn`` / ``error`` / ``off``), two short
lines for the block, its mean time per line and the detail rows, note and actions shown when
it is selected. Pure data (no Qt): the widget (``widgets/pipeline_view.py``) only draws it.
"""
from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: block -> profiler stages (``Snapshot.stage_ms``) whose mean ms it shows
STAGES = {
    'dark': ('dark', 'hot_pixel'),
    'smile': ('orient_smile',),
    'binning': ('bin', 'smile_bin', 'preprocess', 'gpu_preprocess'),
    'flatfield': ('flatfield',),
    'reflectance': ('reflectance',),
    'spatial_bin': ('spatial_bin',),
    'model': ('predict',),
}
#: stages that do more than their block's step (fused / GPU backends run the chain in one go)
_FUSED = {'smile_bin': 'incl. Flip + Smile', 'preprocess': 'incl. Dark … Smile',
          'gpu_preprocess': 'incl. Dark … Smile (GPU)'}

STATES = ('ok', 'warn', 'error', 'off')
LOW_SIGNAL = 0.05                                  # peak below 5 % of full scale: Sensor 'check'


@dataclass
class Block:
    key: str
    title: str
    state: str                                     # ok | warn | error | off
    line1: str = ''
    line2: str = ''
    ms: float | None = None                        # mean time per line (None: not measured)
    rows: list = field(default_factory=list)       # [(label, value)] for the detail panel
    note: str = ''                                 # what to do about a warn / error
    actions: list = field(default_factory=list)    # [(action id, button text)]
    locked: bool = False                           # always on (binning)


def _short(label) -> str:
    """Profile id of a ``.../<id>/flatfield.json`` label, else the label."""
    return Path(str(label)).parent.name or str(label)


def flatfield_state(pipeline) -> tuple[str, str, str] | None:
    """How the flat-field relates to the reflectance factors in use.

    ``None`` while Reflectance is off. Else ``(state, profile in the factors, loaded profile)``
    with state ``'in'`` (the factors contain it: active though its own step is off), ``'out'``
    (it is taken out of them: :meth:`Pipeline.set_flatfield_in_reflectance`), ``'mismatch'``
    (they contain another profile than the one loaded) or ``'absent'``.

    The loaded profile is compared by its values (hsi ``check_flatfield_fits``: the factors were
    fitted with it), not by its folder name; only when neither it fits nor the profile inside
    is known is the name all there is.
    """
    if not pipeline.effective_steps().get('reflectance'):
        return None
    plan = pipeline.plans[0]
    inside = getattr(plan, 'reflectance_flatfield', None)
    if not inside:
        return 'absent', '', ''
    if not getattr(pipeline, 'flatfield_in_reflectance', True) and             getattr(plan, 'reflectance_factors_flat_out', None) is not None:
        return 'out', _short(inside), ''
    loaded = getattr(plan, 'flatfield_profile', None)
    if loaded is not None and not loaded_profile_fits(plan):
        from flatfield_profile import reflectance_flat
        known = reflectance_flat(plan)[0] is not None
        if known or _short(loaded.label) != _short(inside):
            return 'mismatch', _short(inside), _short(loaded.label)
    return 'in', _short(inside), ''


def loaded_profile_fits(plan) -> bool:
    """True when the reflectance factors were fitted with the loaded flat-field profile (cached)."""
    factors = getattr(plan, 'reflectance_factors', None)
    loaded = getattr(plan, 'flatfield_profile', None)
    key = (id(factors), id(loaded))
    cached = getattr(plan, '_loaded_fits_cache', None)
    if cached is None or cached[0] != key:
        from flatfield_profile import flatfield_fits
        cached = (key, loaded is not None and factors is not None and flatfield_fits(factors, loaded.values))
        plan._loaded_fits_cache = cached
    return cached[1]


def input_text(pipeline) -> str:
    """What the model receives now: counts or reflectance, with or without flat-field."""
    steps = pipeline.effective_steps()
    state = flatfield_state(pipeline)
    if state is None:
        return 'Counts + flat-field' if steps.get('flatfield') else 'Counts'
    return 'Reflectance + flat-field' if state[0] in ('in', 'mismatch') else 'Reflectance'


def _ms(snapshot, key) -> tuple[float | None, str]:
    stage_ms = getattr(snapshot, 'stage_ms', None) or {}
    names = [n for n in STAGES.get(key, ()) if n in stage_ms]
    if not names:
        return None, ''
    fused = next((_FUSED[n] for n in names if n in _FUSED), '')
    return float(sum(stage_ms[n] for n in names)), fused


def _prov(plan, key) -> str:
    value = plan.provenance.get(key)
    return '' if value is None else str(value[0])


def _info(plan, key) -> dict:
    return plan.steps.get(key, {'available': False, 'on': False, 'why': ''})


def _num(value, fmt='{:g}', unit='') -> str:
    if value is None:
        return 'n/a'
    try:
        return fmt.format(float(value)) + unit
    except (TypeError, ValueError):
        return str(value)


def _sensor(pipeline, snapshot, needed_fps, live) -> Block:
    plan, src = pipeline.plans[0], pipeline.sources[0]
    meta = src.metadata or {}
    b = Block('sensor', 'Sensor', 'ok', actions=[('sensor', 'Sensor settings…')] if live else [])
    camera = plan.camera_id or (f"{plan.camera_backend} camera" if plan.camera_backend else 'unknown camera')
    b.rows.append(('Camera', camera + (' · live' if live else f" · file {src.name}")))
    b.rows.append(('Exposure', _num(plan.exposure_ms, '{:g}', ' ms')))
    b.rows.append(('Gain', _num(plan.gain)))
    b.rows.append(('Sensor temperature', _num(plan.temperature_c, '{:.1f}', ' °C') + (' (at start)' if live else '')))
    cooling = getattr(src, 'cooling', None) or {}
    if cooling.get('error'):
        b.rows.append(('Cooler target', f"not applied: {cooling['error']}"))
    elif meta.get('target_temperature_c') is not None:
        state = f" · {meta['cooling_state']}" if meta.get('cooling_state') else ''
        b.rows.append(('Cooler target', _num(meta['target_temperature_c'], '{:.1f}', ' °C') + state))
    b.rows.append(('Bin / frame', f"bin {plan.camera_bin} · {src.frame_shape[0]} × {src.frame_shape[1]} · {src.dtype}"))
    b.line1 = f"{_num(plan.exposure_ms, '{:g}', ' ms')} · gain {_num(plan.gain)}"
    fps = None
    if snapshot is not None:
        fps = snapshot.camera_fps
        delivered = max(1, snapshot.frames_delivered + snapshot.lines_lost)
        lost_pct = 100.0 * snapshot.lines_lost / delivered
        need = f" · needed {needed_fps:g} for the belt" if needed_fps else ''
        b.rows.append(('Line rate', f"{fps:.0f} fps measured{need}"))
        b.rows.append(('Lost lines', f"{lost_pct:.2f} % ({snapshot.lines_lost}) · buffers "
                                     f"{snapshot.buffers_in_use} / {snapshot.buffer_frames}"))
        if live and fps < 1:
            b.state, b.note = 'error', 'No frames from the camera: check the cable, trigger and exposure.'
        elif lost_pct > 0.5:
            b.state, b.note = 'warn', 'Lines are lost: the processing cannot keep up (Diagnostics tab).'
        elif needed_fps and fps < 0.95 * needed_fps:
            b.state = 'warn'
            b.note = (f"The camera runs at {fps:.0f} lines/s; {needed_fps:g} are needed for square pixels "
                      "on the belt: shorten the exposure or reduce the frame height.")
        full = meta.get('full_scale') or meta.get('saturation_level')
        frame = snapshot.raw_frame
        if full and frame is not None and frame.size:
            peak = float(np.max(frame))
            clipped = int(np.count_nonzero(frame >= float(full)))
            b.rows.append(('Peak signal', f"{100.0 * peak / float(full):.0f} % of full scale"
                                          + (f" · {clipped} clipped pixels" if clipped else '')))
            if clipped and b.state == 'ok':
                b.state, b.note = 'warn', 'Pixels at full scale: shorten the exposure or lower the gain.'
            elif not clipped and peak < LOW_SIGNAL * float(full) and b.state != 'error':
                low = (f"Peak signal is {100.0 * peak / float(full):.0f} % of full scale: the spectra are mostly "
                       "noise. Lengthen the exposure, raise the gain or add light.")
                b.state, b.note = 'warn', ' '.join(x for x in (low, b.note) if x)
    temp = _num(plan.temperature_c, '{:.1f}', ' °C') if plan.temperature_c is not None else ''
    b.line2 = ' · '.join(x for x in (f"{fps:.0f} fps" if fps is not None else '', temp) if x)
    return b


def _switch_block(pipeline, key, title, on_line, actions=()) -> Block:
    plan = pipeline.plans[0]
    info = _info(plan, key)
    on = pipeline.effective_steps().get(key, False)
    b = Block(key, title, 'ok' if on else 'off', rows=[('State', 'on' if on else 'off'), ('Source', info['why'])])
    b.line1 = on_line if on else ('off' if info['available'] else 'not available')
    if info['available']:
        b.actions.append(('toggle:' + key, 'Switch off' if on else 'Switch on'))
    b.actions.extend(actions)
    return b


def _dark(pipeline) -> Block:
    plan = pipeline.plans[0]
    info = _info(plan, 'dark')
    on = pipeline.effective_steps().get('dark', False)
    label = _prov(plan, 'dark') or info['why']
    b = _switch_block(pipeline, 'dark', 'Dark + hot px', 'on')
    b.rows = [('State', 'on' if on else 'off'), ('Dark file', label or 'none')]
    hot = getattr(plan, 'hot_map', None)
    if hot is not None:
        b.rows.append(('Hot pixels repaired', str(int(np.count_nonzero(hot)))))
    if on:
        name = Path(str(label).split(' ')[0]).name if label else ''
        b.line1 = name or 'on'
        b.line2 = f"{int(np.count_nonzero(hot))} hot px" if hot is not None else ''
    elif plan.stage == 'raw':
        b.state = 'warn' if info['available'] else 'error'
        b.note = ('Dark is off: the dark offset stays in the data and in the reflectance.'
                  if info['available'] else
                  f"No dark for these camera values: record one in hsi. {info['why']}")
    return b


def _smile(pipeline) -> Block:
    plan = pipeline.plans[0]
    steps = pipeline.effective_steps()
    smile, keystone = _info(plan, 'smile'), _info(plan, 'keystone')
    on = steps.get('smile', False)
    b = Block('smile', 'Smile + keystone', 'ok' if on else 'off')
    b.rows = [('Smile', ('on · ' if on else 'off · ') + (_prov(plan, 'smile') or smile['why'] or 'none')),
              ('Keystone', ('on' if steps.get('keystone') else 'off') + (f" · {keystone['why']}" if keystone['why'] else ''))]
    crop = tuple(getattr(plan, 'keystone_crop', (0, 0)) or (0, 0))
    if _prov(plan, 'slit rows'):
        b.rows.append(('Slit rows to model', _prov(plan, 'slit rows')))
    b.line1 = (Path(str(getattr(plan, 'smile_path', '') or '')).stem or 'on') if on else \
        ('off' if smile['available'] else 'not available')
    b.line2 = f"crop {crop[0]} / {crop[1]} rows" if any(crop) else ''
    for key, text in (('smile', 'Smile'), ('keystone', 'Keystone')):
        if _info(plan, key)['available']:
            b.actions.append(('toggle:' + key, f"{text} {'off' if steps.get(key) else 'on'}"))
    if not on and smile['available'] and plan.stage == 'raw':
        b.state, b.note = 'warn', 'Smile is off: the bands of the slit ends are shifted against the centre.'
    return b


def _binning(pipeline) -> Block:
    plan = pipeline.plans[0]
    wl = plan.wavelengths
    b = Block('binning', 'Binning', 'ok', locked=plan.stage == 'raw')
    if wl is not None and len(wl):
        step = float(np.median(np.diff(wl))) if len(wl) > 1 else 0.0
        b.line1 = f"{step:.3g} nm · {len(wl)} bands" if step else f"{len(wl)} bands"
        b.line2 = f"{wl[0]:.0f}–{wl[-1]:.0f} nm"
    b.rows = [('Bins', _prov(plan, 'binning') or _info(plan, 'binning')['why'] or 'none (input already binned)'),
              ('Wavelength', _prov(plan, 'wavelength m,c') or _prov(plan, 'wavelength') or 'n/a')]
    if _prov(plan, 'model bands'):
        b.rows.append(('Model uses', _prov(plan, 'model bands')))
    if plan.stage != 'raw':
        b.state = 'off'
        b.rows.insert(0, ('Input', _prov(plan, 'input') or 'already binned'))
    return b


def _flatfield(pipeline) -> Block:
    plan = pipeline.plans[0]
    info = _info(plan, 'flatfield')
    loaded = getattr(plan, 'flatfield_profile', None)
    state = flatfield_state(pipeline)
    b = Block('flatfield', 'Flat-field', 'off')
    b.rows = [('Loaded profile', _short(loaded.label) if loaded is not None else 'none')]
    if state is not None and state[0] != 'absent':
        from flatfield_profile import reflectance_flat
        flat, _, where = reflectance_flat(plan)
        b.rows.append(('Profile in factors', f"{state[1]} · " + (where if flat is not None else
                                                                 f"values not known: {where}")))
    if loaded is not None:
        b.rows.append(('Needs', ', '.join(f"{k.capitalize()} {'on' if getattr(loaded, k, False) else 'off'}"
                                          for k in ('dark', 'flip', 'smile', 'keystone'))))
    if state is not None and state[0] == 'out':
        b.state, b.line1, b.line2 = 'ok', 'taken out', state[1]
        b.rows.insert(0, ('Applied', 'no: taken out of the reflectance factors (factors × F / F_ROI: the white '
                                     'target stays at its reflectance, the falloff along the slit is left in)'))
        b.actions.append(('flatfield_in:on', 'Put back into the factors'))
    elif state is not None and state[0] in ('in', 'mismatch'):
        b.state, b.line1, b.line2 = 'ok', 'in reference', state[1]
        b.rows.insert(0, ('Applied', f"inside the reflectance factors ({state[1]})"))
        if state[0] == 'mismatch':
            b.state = 'warn'
            b.note = (f"The factors were not fitted with the loaded {state[2]} (they contain {state[1]}): "
                      "calibrate Reflectance again with Flat-field + reference.")
        from flatfield_profile import reflectance_flat
        if reflectance_flat(plan)[0] is not None:
            b.actions.append(('flatfield_in:off', 'Take out of the factors'))
    elif pipeline.effective_steps().get('flatfield'):
        b.state, b.line1, b.line2 = 'ok', 'applied', _short(loaded.label) if loaded is not None else ''
        b.rows.insert(0, ('Applied', 'yes, on counts'))
        b.actions.append(('toggle:flatfield', 'Switch off'))
    elif loaded is not None:
        b.state, b.line1, b.line2 = 'warn', 'not applied', _short(loaded.label)
        if state is not None:                                   # Reflectance without flat-field
            b.rows.insert(0, ('Applied', 'no: the reflectance factors have no flat-field'))
            b.note = ('A flat-field profile is loaded but not used: calibrate Reflectance with '
                      "'Flat-field + reference' to apply both.")
            b.actions.append(('reflectance', 'Calibrate with flat-field…'))
        else:
            b.rows.insert(0, ('Applied', 'no'))
            b.note = 'A flat-field profile is loaded but switched off.'
            if info['available']:
                b.actions.append(('toggle:flatfield', 'Apply alone'))
    else:
        b.line1 = 'no profile'
        b.rows.insert(0, ('Applied', 'no'))
        b.rows.append(('Why', info['why']))
    return b


def _reflectance(pipeline) -> Block:
    plan = pipeline.plans[0]
    info = _info(plan, 'reflectance')
    on = pipeline.effective_steps().get('reflectance', False)
    classifier = pipeline.classifier
    wants = getattr(classifier, 'input_units', 'any') == 'reflectance'
    b = Block('reflectance', 'Reflectance', 'ok' if on else 'off', actions=[('reflectance', 'Calibrate…')])
    why = _prov(plan, 'reflectance') or info['why']
    factors = getattr(plan, 'reflectance_factors', None)
    kind = ''
    if factors is not None:
        taken_out = (not getattr(pipeline, 'flatfield_in_reflectance', True)
                     and getattr(plan, 'reflectance_factors_flat_out', None) is not None)
        kind = ('ref, flat-field out' if getattr(plan, 'reflectance_flatfield', None) and taken_out else
                'flat-field + ref' if getattr(plan, 'reflectance_flatfield', None) else
                'row-wise' if factors.ndim == 2 else 'per band')
    b.rows = [('State', 'on' if on else ('off' if info['available'] else 'not available')),
              ('Kind', kind or 'none'), ('Factors', why or 'none')]
    hsi_note = getattr(plan, '_hsi_reflectance_note', '')
    if hsi_note:
        b.rows.append(('Not used', hsi_note))
    warnings = [w.strip() for w in why.split('WARNING')[1:]] if 'WARNING' in why else []
    b.line1 = kind if on else ('off' if info['available'] else 'not calibrated')
    if on:
        b.line2 = 'check: ' + warnings[0].split(',')[0][:24] if warnings else ''
        if warnings:
            b.state = 'warn'
            b.note = 'Calibrate again: ' + '; '.join(w.rstrip(', ') for w in warnings)
        if getattr(plan, 'reflectance_placeholder', False):
            b.state, b.note = 'warn', 'Placeholder factors: speed is real, reflectance values are not.'
        b.actions.append(('toggle:reflectance', 'Switch off'))
    else:
        stored = stored_reflectance(plan)
        if stored is not None:
            b.line1 = 'file not usable'
            b.rows.append(('Stored file', f"{stored[0]}: {stored[1]}"))
        if info['available']:
            b.actions.append(('toggle:reflectance', 'Switch on'))
        if wants:
            b.state = 'error'
            b.note = 'The model expects reflectance but receives counts: calibrate or switch Reflectance on.'
            if stored is not None:
                b.note = (f"The model expects reflectance but receives counts: the stored {stored[0]} does not fit "
                          f"this input ({stored[1]}). Calibrate again.")
    return b


def _spatial_bin(pipeline) -> Block:
    n = pipeline.spatial_bin
    b = Block('spatial_bin', '2×2 bin', 'ok' if n == 2 else 'off',
              line1='on' if n == 2 else 'off', line2='2 × 2 pixels' if n == 2 else '1 × 1 pixels')
    b.rows = [('State', 'mean of 2 × 2 pixels before the model' if n == 2 else 'off: every pixel classified')]
    b.actions = [('spatial_bin', 'Switch off' if n == 2 else 'Switch on')]
    return b


def _model(pipeline) -> Block:
    plan = pipeline.plans[0]
    classifier = pipeline.classifier
    units = getattr(classifier, 'input_units', 'any')
    receives = input_text(pipeline)
    name = Path(str(getattr(classifier, 'path', '') or 'model')).name
    classes = getattr(classifier, 'category_names', {}) or {}
    b = Block('model', 'Model', 'ok', line1=name, line2=f"{len(classes)} classes" if classes else '',
              actions=[('model', 'Choose model…')])
    b.rows = [('Model', name), ('Expects', units), ('Receives', receives),
              ('Bands', _prov(plan, 'model bands') or 'n/a'),
              ('Classes', ', '.join(str(v) for v in classes.values()) or 'n/a')]
    if units != 'any' and (units == 'reflectance') != receives.startswith('Reflectance'):
        b.state = 'error'
        b.note = f"The model was trained on {units}, it receives {receives.lower()}: the classes will be wrong."
    return b


def describe(pipeline, snapshot=None, needed_fps: float | None = None, live: bool = False):
    """``(headline, blocks)`` of the pipeline now (first file / the live camera)."""
    plan = pipeline.plans[0]
    steps = pipeline.effective_steps()
    blocks = [_sensor(pipeline, snapshot, needed_fps, live), _dark(pipeline),
              _switch_block(pipeline, 'flip', 'Flip', 'slit reversed'), _smile(pipeline), _binning(pipeline),
              _flatfield(pipeline), _reflectance(pipeline), _spatial_bin(pipeline), _model(pipeline)]
    if plan.stage != 'raw':
        for b in blocks[1:4]:
            b.state, b.line1, b.line2, b.actions, b.note = 'off', 'done in file', '', [], ''
    if not steps.get('flip') and plan.stage == 'raw':
        blocks[2].line2 = ''
    for b in blocks:
        b.ms, fused = _ms(snapshot, b.key)
        if fused:
            b.rows.append(('Time', f"{b.ms:.2f} ms per line, {fused}"))
    return input_text(pipeline), blocks


# ------------------------------------------------------------------ status bar and files ---

def status_items(pipeline, snapshot=None, live: bool = False, temperature: float | None = None) -> list:
    """``[(text, state)]`` for the Setup tab's status bar: camera, line rate, lost lines, temperature."""
    plan, src = pipeline.plans[0], pipeline.sources[0]
    items = [((plan.camera_id or plan.camera_backend or 'camera') + (' · live' if live else f" · file {src.name}"),
              'neutral')]
    if snapshot is not None:
        items.append((f"{snapshot.camera_fps:.0f} fps", 'error' if live and snapshot.camera_fps < 1 else 'neutral'))
        delivered = snapshot.frames_delivered + snapshot.lines_lost
        lost = 100.0 * snapshot.lines_lost / delivered if delivered else 0.0
        items.append((f"{lost:.2f} % lost", 'warn' if lost > 0.5 else 'neutral'))
    temp = temperature if temperature is not None else plan.temperature_c
    if temp is not None:
        made = reflectance_temperature(plan)
        limit = getattr(plan, 'reflectance_max_dtemp_c', 0.0) or 0.0
        drift = made is not None and limit and abs(temp - made) > limit
        items.append((f"{temp:.1f} °C" + (' (at start)' if live and temperature is None else ''),
                      'warn' if drift else 'neutral'))
    return items


def reflectance_temperature(plan) -> float | None:
    """Sensor temperature the reflectance factors in use were made at (``None``: unknown)."""
    return getattr(plan, 'reflectance_temperature_c', None)


_STAMP = re.compile(r'(20\d{6})[_-]?(\d{6})')


def _age_h(text, now: datetime.datetime | None = None) -> float | None:
    """Hours since a ``YYYYMMDD_HHMMSS`` stamp in ``text``, else since an existing file's change."""
    now = now or datetime.datetime.now()
    match = _STAMP.search(str(text))
    if match:
        try:
            made = datetime.datetime.strptime(match.group(1) + match.group(2), '%Y%m%d%H%M%S')
            return max(0.0, (now - made).total_seconds() / 3600.0)
        except ValueError:
            pass
    for token in str(text).replace('(', ' ').replace(')', ' ').replace(';', ' ').split():
        path = Path(token.strip(',[]'))
        if path.suffix in ('.npz', '.npy', '.json', '.h5') and path.is_file():
            return max(0.0, (now.timestamp() - os.path.getmtime(path)) / 3600.0)
    return None


def _age_text(hours: float | None) -> str:
    if hours is None:
        return ''
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.0f} h"
    return f"{hours / 24:.0f} d"


def _file_name(text: str) -> str:
    """``dark_b2_e4.npz`` of ``path/dark_b2_e4.npz (37 hot pixels)``; ``none`` for an unused one."""
    text = str(text or '').strip()
    if not text or text.startswith(('none', 'white reference not', 'not usable')):
        return 'none'
    name = Path(text.split(' (')[0].split(' - ')[0].split(';')[0].split(' ')[0]).name
    return name if len(name) <= 40 else name[:38] + '…'


def calibration_files(pipeline) -> list:
    """``[(step, file, age text, state)]`` of the calibration files the pipeline uses now."""
    plan = pipeline.plans[0]
    steps = pipeline.effective_steps()
    rows = []
    if plan.stage == 'raw':
        dark = _prov(plan, 'dark')
        rows.append(('Dark', _file_name(dark), _age_text(_age_h(dark)), 'ok' if steps.get('dark') else 'error'))
        smile = getattr(plan, 'smile_path', None)
        rows.append(('Smile', Path(smile).name if smile else 'none', _age_text(_age_h(smile or '')),
                     'ok' if steps.get('smile') else 'warn' if smile else 'off'))
        wl = plan.provenance.get('wavelength m,c')
        rows.append(('Wavelength', _file_name(wl[1]) if wl else 'none', _age_text(_age_h(wl[1] if wl else '')),
                     'ok' if wl else 'warn'))
    flat = getattr(plan, 'flatfield_profile', None)
    state = flatfield_state(pipeline)
    flat_used = steps.get('flatfield') or (state is not None and state[0] == 'in')
    rows.append(('Flat-field', _short(flat.label) if flat is not None else 'none',
                 _age_text(_age_h(flat.label)) if flat is not None else '',
                 'ok' if flat_used else 'warn' if flat is not None else 'off'))
    refl = _prov(plan, 'reflectance')
    on = steps.get('reflectance')
    stored = stored_reflectance(plan)
    if not on and stored is not None:                  # a file is there but does not fit this input
        rows.append(('Reflectance', f"{stored[0]} · not usable", '', 'warn'))
        return rows
    age = _age_h(refl)
    limit = getattr(plan, 'reflectance_max_age_h', 0.0) or 0.0
    rows.append(('Reflectance', _file_name(refl), _age_text(age),
                 'warn' if on and ('WARNING' in refl or (limit and age is not None and age > limit)) else
                 'ok' if on else 'off'))
    return rows


def stored_reflectance(plan) -> tuple[str, str] | None:
    """``(file name, why)`` of a reflectance file that exists but could not be used (else ``None``)."""
    if not str(_prov(plan, 'reflectance')).startswith('not usable'):
        return None
    why = str(_info(plan, 'reflectance').get('why', ''))
    match = re.search(r'([\w.\-]+\.npy):\s*(.*)', why)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def next_steps(blocks) -> list:
    """``[(state, text, action id, button text)]``: what to fix, blocking steps first.

    One line per block in ``error`` / ``warn`` with a note. A model that only blocks because it
    receives counts is folded into the Reflectance line (calibrating fixes both).
    """
    by_key = {b.key: b for b in blocks}
    refl, model = by_key.get('reflectance'), by_key.get('model')
    folded = (refl is not None and model is not None and refl.state == 'error' and model.state == 'error'
              and 'counts' in model.note)
    items = []
    for state in ('error', 'warn'):
        for b in blocks:
            if b.state != state or not b.note or (folded and b.key == 'model'):
                continue
            text = f"{b.title}: {b.note}"
            if folded and b.key == 'reflectance':
                text += ' This also clears the Model block.'
            action = next(((a, t) for a, t in b.actions if not a.startswith('toggle:')), None) \
                or (b.actions[0] if b.actions else ('', ''))
            items.append((state, text, *action))
    return items


def slit_profiles(pipeline, snapshot=None) -> tuple[list, str]:
    """Brightness along the slit (mean over bands, 1 = brightest row) for the profile plot.

    ``([(label, rows, values, colour token, dashed)], note)``. The loaded flat-field profile is
    the optics' response measured on a white when it was made; with it, the response left after
    the correction in use. With ``snapshot`` (Check with white): what the model receives now,
    which only means something with a white target under the slit.
    """
    plan = pipeline.plans[0]
    series, notes = [], []
    flat = getattr(plan, 'flatfield_profile', None)
    steps = pipeline.effective_steps()
    state = flatfield_state(pipeline)
    if flat is not None:
        values = np.asarray(flat.values, np.float64)
        response = values.mean(axis=1)
        rows = np.arange(response.size)
        series.append(('optics (flat-field profile)', rows, list(response / (np.nanmax(response) or 1.0)),
                       'warning', False))
        after = response.copy()
        if state is not None and state[0] == 'out':
            pass                                                   # taken out: the falloff stays
        elif steps.get('flatfield') or (state is not None and state[0] in ('in', 'mismatch')):
            after = (values / values).mean(axis=1)                 # divided by the profile itself
        elif steps.get('reflectance') and getattr(plan, 'reflectance_factors', None) is not None \
                and plan.reflectance_factors.ndim == 2 and plan.reflectance_factors.shape[0] == after.size:
            after = (values * np.asarray(plan.reflectance_factors, np.float64)).mean(axis=1)
        series.append(('with the correction in use', rows, list(after / (np.nanmax(after) or 1.0)),
                       'success', True))
        notes.append(f"Darkest row vs brightest: optics {_spread(response):.0f} % darker, "
                     f"after the correction in use {_spread(after):.0f} %")
    if snapshot is not None and getattr(snapshot, 'raw_canvas', None) is not None and snapshot.raw_canvas.size:
        with np.errstate(all='ignore'):
            live = np.nanmean(snapshot.raw_canvas, axis=1)
        if np.isfinite(live).any() and np.nanmax(live) > 0:
            series.append(('model input now', np.arange(live.size), list(live / np.nanmax(live)), 'accent', False))
            notes.append(f"model input now {_spread(live):.0f} % (meaningful only with a white under the slit)")
    if not series:
        notes.append('No flat-field profile is loaded for this camera. Put a white target under the slit and '
                     'press Check with white to see the brightness along the slit.')
    return series, ' · '.join(notes)


def _spread(values) -> float:
    """How much darker the darkest of the inner 90 % of the rows is than the brightest, in %."""
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return 0.0
    k = max(1, v.size // 20)
    inner = v[k:-k] if v.size > 2 * k + 2 else v
    top = inner.max()
    return 100.0 * (1.0 - inner.min() / top) if top > 0 else 0.0
