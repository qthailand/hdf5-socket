# -*- coding: utf-8 -*-
"""
Convert an ENVI cube (BIP; BIL and BSQ also work) back to an HDF5 spectrogram by a plain
transpose - the reverse of ``convert_h5.py``. The cube is (lines, samples, bands) =
(slit, frame, spectral); the HDF5 dataset ``spectrogram`` is (frame, slit, spectral),
gzip level 4 with one frame per chunk, as HSI writes it.

Header fields become root attributes (``camera bin`` -> ``camera_bin``), so a BIP written by
``convert_h5.py`` gives back the original attributes. A cube without ``c nm`` but with a
``wavelength`` list (e.g. a corrected, binned HSI cube) also gets ``wavelength_nm``, which
``sources.py`` reads as stage ``'binned'``.

    python convert_bip.py data/first.bip data/first_from_bip.h5
"""
import argparse
import re
from pathlib import Path

import h5py
import numpy as np

BLOCK_FRAMES = 64      # frames read and written per step
ENVI_DTYPES = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64,
               12: np.uint16, 13: np.uint32}
# ENVI fields that describe the file layout, not the recording: never copied to attributes
LAYOUT_FIELDS = {'description', 'data type', 'byte order', 'interleave', 'lines', 'samples', 'bands',
                 'header offset', 'file type', 'wavelength', 'wavelength units'}


def _scalar(text):
    """Header value -> int/float/bool/str."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {'True': True, 'False': False}.get(text, text)


def read_header(hdr_path):
    """ENVI header -> dict with lower-case keys; ``{a, b}`` lists become Python lists."""
    text = Path(hdr_path).read_text(encoding='utf-8', errors='replace')
    if not text.lstrip().startswith('ENVI'):
        raise ValueError(f"{hdr_path}: not an ENVI header")
    fields = {}
    for match in re.finditer(r'^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)', text, re.MULTILINE):
        key, value = match.group(1).strip().lower(), match.group(2).strip()
        if value.startswith('{'):
            inner = value[1:-1].strip()
            value = inner if key == 'description' else [_scalar(v.strip()) for v in inner.split(',') if v.strip()]
        else:
            value = _scalar(value)
        fields[key] = value
    return fields


def _header_path(data_path):
    for candidate in (Path(str(data_path) + '.hdr'), Path(data_path).with_suffix('.hdr')):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No ENVI header for {data_path}")


def convert(bip_path, h5_path, dataset='spectrogram', overwrite=False):
    bip_path, h5_path = Path(bip_path), Path(h5_path)
    if bip_path.suffix.lower() == '.hdr':
        bip_path = bip_path.with_suffix('') if bip_path.with_suffix('').is_file() else bip_path
    if h5_path.exists() and not overwrite:
        raise FileExistsError(f"{h5_path} exists (use --overwrite or another output name)")
    header = read_header(_header_path(bip_path))
    lines, samples, bands = int(header['lines']), int(header['samples']), int(header['bands'])
    dtype = np.dtype(ENVI_DTYPES[int(header.get('data type', 4))])
    dtype = dtype.newbyteorder('>' if int(header.get('byte order', 0)) == 1 else '<')
    interleave = str(header.get('interleave', 'bsq')).lower()
    shapes = {'bip': (lines, samples, bands), 'bil': (lines, bands, samples), 'bsq': (bands, lines, samples)}
    to_lsb = {'bip': (0, 1, 2), 'bil': (0, 2, 1), 'bsq': (1, 2, 0)}      # file axes -> (lines, samples, bands)
    if interleave not in shapes:
        raise ValueError(f"{bip_path.name}: unsupported interleave {interleave!r}")

    cube = np.memmap(bip_path, dtype=dtype, mode='r', offset=int(header.get('header offset', 0)),
                     shape=shapes[interleave]).transpose(to_lsb[interleave])      # (slit, frame, spectral)

    attrs = {key.replace(' ', '_'): value for key, value in header.items() if key not in LAYOUT_FIELDS}
    if 'c_nm' not in attrs and isinstance(header.get('wavelength'), list):
        attrs['wavelength_nm'] = np.asarray(header['wavelength'], np.float64)
    attrs.setdefault('axes', 'frame,slit,spectral')
    attrs['shape'] = [samples, lines, bands]
    if 'frame_height' in attrs or 'frame_width' in attrs:      # a cropped cube has smaller frames
        attrs['frame_height'], attrs['frame_width'] = lines, bands
    attrs['dtype'] = dtype.newbyteorder('=').name

    with h5py.File(h5_path, 'w') as f:
        for key, value in attrs.items():
            f.attrs[key] = value
        ds = f.create_dataset(dataset, shape=(samples, lines, bands), dtype=dtype.newbyteorder('='),
                              chunks=(1, lines, bands), compression='gzip', compression_opts=4)
        for i in range(0, samples, BLOCK_FRAMES):
            ds[i:i + BLOCK_FRAMES] = cube[:, i:i + BLOCK_FRAMES, :].transpose(1, 0, 2)
            print(f"\r{bip_path.name}: {min(i + BLOCK_FRAMES, samples)}/{samples} frames", end='', flush=True)
    del cube
    print()
    print(f"wrote {h5_path} ({dataset} {(samples, lines, bands)}, {dtype.newbyteorder('=')})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bip', help='input ENVI data file (or its .hdr)')
    parser.add_argument('h5', nargs='?', help='output HDF5 file (default: <bip name>.h5 beside the input)')
    parser.add_argument('--dataset', default='spectrogram', help='HDF5 dataset name')
    parser.add_argument('--overwrite', action='store_true', help='replace an existing output file')
    args = parser.parse_args()
    source = Path(args.bip)
    default = (source.with_suffix('') if source.suffix.lower() == '.hdr' else source).with_suffix('.h5')
    convert(args.bip, args.h5 or default, args.dataset, args.overwrite)
