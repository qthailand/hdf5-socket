"""ENVI header parsing and band reads extracted without changing layout semantics.

Data conventions:
    ENVI ``lines``/``samples``/``bands`` map to ``(slit, scan, wavelength)`` for
    cubes written by this package.  Band reads return float32 ``(lines, samples)``.
    The real-valued data types 1, 2, 3, 4, 5, 12, 13, 14 and 15 (:data:`ENVI_DATA_TYPES`)
    are decoded; complex (6, 9) and unknown codes raise ``ValueError``.  Band reads
    honour the header ``byte order`` and ``header offset`` and read BSQ, BIL and BIP
    files.  Wavelengths are returned in nm (micrometre headers are converted).

File handles:
    Band reads keep the memory maps of the last two data files open (re-mapping a
    BIP file for every band is about 40x slower).  Call :func:`release_band_maps`
    before overwriting or deleting such a file: on Windows a mapped file cannot be
    replaced.  A file that changed on disk (size or modification time) is re-mapped.

Threading:
    Any thread; the map cache is protected by a lock.

Example:
    >>> import os, tempfile
    >>> from modules.cube_io.envi import parse_envi_header
    >>> d = tempfile.mkdtemp(); p = os.path.join(d, "x.img")
    >>> text = os.linesep.join(["ENVI", "lines = 2", "samples = 3", "bands = 4"])
    >>> open(p + ".hdr", "w").write(text) > 0
    True
    >>> m = parse_envi_header(p); (m["lines"], m["samples"], m["bands"], m["wavelength"])
    (2, 3, 4, None)
"""
import os
import re
import threading
from collections import OrderedDict

import numpy as np

#: ENVI ``data type`` code -> numpy dtype (byte order applied separately).
ENVI_DATA_TYPES = {
    1: np.dtype(np.uint8), 2: np.dtype(np.int16), 3: np.dtype(np.int32),
    4: np.dtype(np.float32), 5: np.dtype(np.float64), 12: np.dtype(np.uint16),
    13: np.dtype(np.uint32), 14: np.dtype(np.int64), 15: np.dtype(np.uint64),
}

#: ``wavelength units`` values (lower case) that mean micrometres.
_MICROMETRE_UNITS = ('micrometers', 'micrometres', 'micrometer', 'micrometre', 'microns', 'micron',
                     'um', 'µm', 'μm')


def envi_dtype(code):
    """The numpy dtype of an ENVI ``data type`` code.

    Raises:
        ValueError: complex (6, 9) or unknown codes.
    """
    try:
        return ENVI_DATA_TYPES[int(code)]
    except (KeyError, ValueError, TypeError):
        raise ValueError(f"Unsupported ENVI data type {code!r} (supported: "
                         f"{', '.join(str(c) for c in sorted(ENVI_DATA_TYPES))})") from None


def envi_data_type(dtype):
    """The ENVI ``(data type, byte order)`` of a numpy dtype (byte order 1 = big endian).

    Raises:
        ValueError: the dtype has no ENVI code (bool, int8, float16, complex, ...).
    """
    dtype = np.dtype(dtype)
    for code, known in ENVI_DATA_TYPES.items():
        if dtype.newbyteorder('=') == known:
            big = dtype.byteorder == '>' or (dtype.byteorder == '=' and not np.little_endian)
            return code, int(big)
    raise ValueError(f"dtype {dtype} cannot be written to ENVI (supported: "
                     f"{', '.join(str(d) for d in ENVI_DATA_TYPES.values())})")


_MAX_MAPS = 2
_maps = OrderedDict()            # (path, mtime, size, dtype, offset, shape) -> np.memmap
_maps_lock = threading.Lock()


def _band_map(filepath, dtype, offset, shape):
    """Read-only memory map of a data file, reused while the file is unchanged."""
    path = os.path.abspath(filepath)
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size, dtype.str, offset, shape)
    with _maps_lock:
        data = _maps.get(key)
        if data is None:
            for old in [k for k in _maps if k[0] == path]:       # the file was rewritten
                del _maps[old]
            data = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)
            _maps[key] = data
            while len(_maps) > _MAX_MAPS:
                _maps.popitem(last=False)
        else:
            _maps.move_to_end(key)
        return data


def release_band_maps(filepath=None):
    """Close the cached memory maps of ``filepath`` (or of every file).

    Call it before overwriting or deleting a file that was read with
    :func:`_read_envi_band_float32`.
    """
    with _maps_lock:
        if filepath is None:
            _maps.clear()
        else:
            path = os.path.abspath(filepath)
            for old in [k for k in _maps if k[0] == path]:
                del _maps[old]


def parse_envi_header(filepath):
    """Parse the ENVI header that belongs to a data file.

    Args:
        filepath: Path of the data file; the header is looked up as
            ``filepath + '.hdr'`` and then as ``<filepath without extension>.hdr``.

    Returns:
        dict: Header fields with lower-cased keys and string values, plus
        defaults ``data_type=4``, ``byte_order=0``, ``interleave='bsq'``,
        ``description``; ``lines``/``samples``/``bands`` as ``int``;
        ``wavelength`` as a float32 array in nm or ``None``.  A ``wavelength units``
        of micrometres is converted (values x 1000, ``wavelength units`` becomes
        ``Nanometers``); other units are returned as written.

    Raises:
        ValueError: no header file found, or a wavelength entry is not a number.
        KeyError: the header lacks ``lines``, ``samples`` or ``bands``.
    """
    hdr_path = filepath + '.hdr'
    if not os.path.exists(hdr_path):
        hdr_path = os.path.splitext(filepath)[0] + '.hdr'
    if not os.path.exists(hdr_path):
        raise ValueError(f"Header file not found for {filepath}")

    meta = {'data_type': 4, 'byte_order': 0,
            'interleave': 'bsq', 'description': 'Processed Data'}
    with open(hdr_path, 'rb') as f:
        raw = f.read()
    try:
        txt = raw.decode('utf-8')
    except UnicodeDecodeError:
        txt = raw.decode('latin-1')          # Windows tools write e.g. 'µm' as cp1252
    txt = txt.replace('\r\n', '\n').replace('\r', '\n')

    m_desc = re.search(r'description\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
    if m_desc:
        meta['description'] = m_desc.group(1).strip().replace('\n', ' ')
    else:
        m_desc_simple = re.search(
            r'^description\s*=\s*(.*)$', txt, re.IGNORECASE | re.MULTILINE)
        if m_desc_simple:
            meta['description'] = m_desc_simple.group(1).strip()

    for line in txt.split('\n'):
        if '=' in line and 'description' not in line.lower():
            k, v = line.split('=', 1)
            k = k.strip().lower()
            if k != 'wavelength':
                meta[k] = v.strip()

    m_wl = re.search(r'wavelength\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
    if m_wl:
        vals = m_wl.group(1).replace('\n', ' ').split(',')
        meta['wavelength'] = np.array(
            [float(v.strip()) for v in vals if v.strip() != ''], dtype=np.float32)
        if str(meta.get('wavelength units', '')).strip().lower() in _MICROMETRE_UNITS:
            meta['wavelength'] = meta['wavelength'] * np.float32(1000.0)
            meta['wavelength units'] = 'Nanometers'
    else:
        meta['wavelength'] = None

    meta['lines'] = int(meta['lines'])
    meta['samples'] = int(meta['samples'])
    meta['bands'] = int(meta['bands'])
    return meta


def _read_envi_band_float32(filepath, meta, band_idx):
    """Read one band of an ENVI data file as float32 (public alias ``read_envi_band``).

    Args:
        filepath: Path of the data file (without header).
        meta: Dict from :func:`parse_envi_header` (uses ``lines``, ``samples``,
            ``bands``, ``data type``/``data_type`` and ``interleave``).
        band_idx: Zero-based band (wavelength) index.

    Returns:
        numpy.ndarray: New float32 array ``(lines, samples)``.  The file is memory
        mapped, so only the requested band is read (BSQ, BIL and BIP alike); the map
        stays cached (see :func:`release_band_maps`).  ``byte order`` (1 = big endian) and
        ``header offset`` are honoured.  32/64-bit integers beyond 2**24 lose
        precision in the float32 result.

    Raises:
        ValueError: unsupported data type or interleave, or the file is smaller than
            the header describes.
    """
    L, S, B = meta['lines'], meta['samples'], meta['bands']
    dt = envi_dtype(meta.get('data type', meta.get('data_type', '4')))
    bpp = dt.itemsize
    dt = dt.newbyteorder('>' if int(meta.get('byte order', meta.get('byte_order', 0))) == 1 else '<')
    offset = int(meta.get('header offset', 0))

    interleave = str(meta.get('interleave', 'bsq')).lower().strip()
    shapes = {'bsq': (B, L, S), 'bil': (L, B, S), 'bip': (L, S, B)}
    if interleave not in shapes:
        raise ValueError(f"Unsupported interleave: {interleave}")
    expected = offset + L * S * B * bpp
    size = os.path.getsize(filepath)
    if size < expected:
        raise ValueError(f"{filepath}: file has {size} bytes, the header describes {expected}")

    data = _band_map(filepath, dt, offset, shapes[interleave])
    if interleave == 'bsq':
        band = data[band_idx]
    elif interleave == 'bil':
        band = data[:, band_idx, :]
    else:
        band = data[:, :, band_idx]
    return np.array(band, dtype=np.float32)        # a copy, never a view of the map


