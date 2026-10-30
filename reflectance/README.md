# Reflectance factors, one per data source

The last step before inference: `reflectance = binned counts x factor` (hsi `modules/radiometric`).

Name each profile after the data source it belongs to (file name without extension):

```
data/spectrogram02.h5        ->  reflectance/spectrogram02.npy  (+ spectrogram02.wavelength.npy)
dataset/dataset.bip.hdr      ->  reflectance/dataset.npy        (+ dataset.wavelength.npy)
```

The files are what hsi writes with `save_factor_profile` (Calibration window / `python -m
modules.radiometric --compute rowwise ... --output <name>.npy`):

* `<name>.npy`: row-wise factors `(slit rows, bands)` float32, or per-band factors `(bands,)`.
* `<name>.wavelength.npy`: band centres (nm) of the cube the factors were made from. Strongly
  recommended: with it the bands are checked by wavelength, without it only by count.

The factors must be made on the same bands the simulator bins to (same camera bin, wavelength
calibration and resolution) and, for row-wise factors, the same slit rows and orientation.
