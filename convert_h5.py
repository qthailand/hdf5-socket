# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:49:18 2026

@author: TSC

Convert an HDF5 spectrogram (frame, slit, spectral) to an ENVI BIP cube by a plain
transpose - no dark/hot pixel/flip/Smile/binning. BIP on disk is (lines, samples, bands)
= (slit, frame, spectral), i.e. ``spectrogram.transpose(1, 0, 2)``, the same axes HSI
writes (lines = slit, samples = scan, bands = wavelength).

    python convert_h5.py data/first.h5 dataset/first_raw.bip
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

BLOCK_FRAMES = 64      # frames read and written per step (~170 MB for 1024 x 1280 uint16)
ENVI_DTYPES = {np.dtype(np.uint8): 1, np.dtype(np.int16): 2, np.dtype(np.int32): 3,
               np.dtype(np.float32): 4, np.dtype(np.float64): 5, np.dtype(np.uint16): 12,
               np.dtype(np.uint32): 13}


def _plain(value):
    """HDF5 attribute -> plain Python value (str/int/float/bool/list)."""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _wavelengths(metadata, n_spectral):
    """nm of each spectral pixel: ``m * j + c`` (m at the recorded camera bin), or None."""
    c = metadata.get('c_nm')
    m = metadata.get('m_nm_per_px')
    if m is None and metadata.get('m_nm_per_px_bin1') is not None:
        m = metadata['m_nm_per_px_bin1'] * metadata.get('camera_bin', 1)
    if m is None or c is None:
        return None
    return m * np.arange(n_spectral) + c


def write_header(hdr_path, lines, samples, bands, dtype, metadata, wavelengths):
    fields = [
        'ENVI',
        'description = {Raw spectrogram converted from HDF5 (transpose only, no corrections)}',
        f'data type = {ENVI_DTYPES[dtype]}',
        'byte order = 0',
        'interleave = bip',
        f'lines = {lines}',
        f'samples = {samples}',
        f'bands = {bands}',
    ]
    # every root attribute (convert_bip.py turns them back into attributes)
    for key, value in metadata.items():
        if isinstance(value, list):
            value = '{' + ', '.join(str(v) for v in value) + '}'
        fields.append(f'{key.replace("_", " ")} = {value}')
    if wavelengths is not None:
        fields.append('wavelength units = Nanometers')
        fields.append('wavelength = {' + ', '.join(f'{w:.5f}' for w in wavelengths) + '}')
    Path(hdr_path).write_text('\n'.join(fields) + '\n', encoding='utf-8')


def convert(h5_path, bip_path, dataset='spectrogram'):
    h5_path, bip_path = Path(h5_path), Path(bip_path)
    with h5py.File(h5_path, 'r') as f:
        if dataset not in f:
            raise ValueError(f"{h5_path.name}: no dataset {dataset!r} (found {list(f.keys())})")
        ds = f[dataset]
        if ds.ndim != 3:
            raise ValueError(f"{h5_path.name}:{dataset} must be (frame, slit, spectral), got {ds.shape}")
        metadata = {k: _plain(v) for k, v in f.attrs.items()}
        n_frame, n_slit, n_spectral = ds.shape
        dtype = ds.dtype.newbyteorder('<')

        # BIP = (lines, samples, bands) = (slit, frame, spectral)
        bip = np.memmap(bip_path, dtype=dtype, mode='w+', shape=(n_slit, n_frame, n_spectral))
        for i in range(0, n_frame, BLOCK_FRAMES):
            block = ds[i:i + BLOCK_FRAMES]                       # (frames, slit, spectral)
            bip[:, i:i + block.shape[0], :] = block.transpose(1, 0, 2)
            print(f"\r{h5_path.name}: {min(i + BLOCK_FRAMES, n_frame)}/{n_frame} frames", end='', flush=True)
        bip.flush()
        del bip
        print()

    write_header(Path(str(bip_path) + '.hdr'), n_slit, n_frame, n_spectral, np.dtype(dtype),
                 metadata, _wavelengths(metadata, n_spectral))
    print(f"wrote {bip_path} (lines={n_slit}, samples={n_frame}, bands={n_spectral}, {dtype})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('h5', help='input HDF5 file')
    parser.add_argument('bip', nargs='?', help='output BIP file (default: <h5 name>.bip beside the input)')
    parser.add_argument('--dataset', default='spectrogram', help='HDF5 dataset name')
    args = parser.parse_args()
    convert(args.h5, args.bip or Path(args.h5).with_suffix('.bip'), args.dataset)
