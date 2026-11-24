"""tools/train_sgd_coco.py trains on every labelled cube of a folder: pairing, class merging, checks."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_sgd_coco import (find_cubes, label_path_for, load_training_set,  # noqa: E402
                                  merge_categories, train_smart_bands_sgd)

BANDS = 12
WAVELENGTHS = np.linspace(900.0, 1700.0, BANDS)


def write_cube(folder: Path, name: str, lines=40, samples=30, bands=BANDS, wavelengths=None, seed=0,
               classes=((1, 'Mango'), (2, 'Bg')), image_size=None, label_name=None, dtype=np.float32):
    """ENVI BIP cube ``name.hdr`` + ``name.bip`` and COCO ``label_name`` (default ``name.json``).

    The cube is cut into equal vertical strips, one per entry of ``classes``; each class has its
    own spectrum so the training is separable.
    """
    rng = np.random.default_rng(seed)
    wl = WAVELENGTHS if wavelengths is None else wavelengths
    cube = rng.normal(0.4, 0.02, (lines, samples, bands)).astype(dtype)
    edges = [samples * i // len(classes) for i in range(len(classes) + 1)]
    for k in range(len(classes)):
        cube[:, edges[k]:edges[k + 1], :] += k * np.linspace(0.0, 0.3, bands, dtype=dtype)
    cube.tofile(folder / f"{name}.bip")
    (folder / f"{name}.hdr").write_text(
        "ENVI\n"
        f"samples = {samples}\nlines = {lines}\nbands = {bands}\n"
        "data type = 4\ninterleave = bip\nbyte order = 0\n"
        "wavelength = {" + ", ".join(f"{w:.3f}" for w in wl) + "}\n", encoding='utf-8')
    boxes = {cid: [edges[k], 0, edges[k + 1] - 1, 0, edges[k + 1] - 1, lines - 1, edges[k], lines - 1]
             for k, (cid, _) in enumerate(classes)}
    w, h = image_size or (samples, lines)
    coco = {'images': [{'id': 1, 'width': w, 'height': h, 'file_name': f"{name}.hdr"}],
            'categories': [{'id': cid, 'name': cname} for cid, cname in classes],
            'annotations': [{'id': k + 1, 'image_id': 1, 'category_id': cid, 'segmentation': [pts]}
                            for k, (cid, pts) in enumerate(boxes.items())]}
    (folder / (label_name or f"{name}.json")).write_text(json.dumps(coco), encoding='utf-8')


def test_label_path_for_finds_the_coco_file_by_either_naming(tmp_path):
    for hdr, label in (('a.hdr', 'a.json'), ('b.hdr', 'b.bip.json'), ('c.bip.hdr', 'c.bip.json'),
                       ('d.bip.hdr', 'd.json')):
        (tmp_path / hdr).write_text('ENVI')
        (tmp_path / label).write_text('{}')
        assert label_path_for(tmp_path / hdr) == tmp_path / label
    (tmp_path / 'lonely.hdr').write_text('ENVI')
    assert label_path_for(tmp_path / 'lonely.hdr') is None


def test_find_cubes_lists_labelled_headers_and_skips_the_rest(tmp_path):
    write_cube(tmp_path, 'one')
    write_cube(tmp_path, 'two', seed=1, label_name='two.bip.json')
    (tmp_path / 'unlabelled.hdr').write_text('ENVI')
    cubes, skipped = find_cubes(tmp_path)
    assert [h.name for h, _ in cubes] == ['one.hdr', 'two.hdr']
    assert [h.name for h in skipped] == ['unlabelled.hdr']
    with pytest.raises(ValueError, match='not found'):
        find_cubes(tmp_path / 'missing')


def test_merge_keeps_ids_that_agree_and_matches_by_name():
    names, luts = merge_categories([{1: 'Mango', 2: 'Bg'}, {1: 'mango ', 2: 'BG', 3: 'Kratom'}])
    assert names == {1: 'Mango', 2: 'Bg', 3: 'Kratom'}
    assert luts[0][1] == 1 and luts[1][3] == 3


def test_merge_renumbers_when_the_ids_disagree():
    names, luts = merge_categories([{1: 'Mango', 2: 'Bg'}, {1: 'Kratom', 2: 'Mango'}])
    assert names == {1: 'Mango', 2: 'Bg', 3: 'Kratom'}
    assert [luts[0][1], luts[0][2]] == [1, 2]
    assert [luts[1][1], luts[1][2]] == [3, 1]                    # Kratom -> 3, Mango -> 1


def test_load_pools_the_pixels_of_every_cube(tmp_path):
    write_cube(tmp_path, 'one', lines=40, samples=30)
    write_cube(tmp_path, 'two', lines=20, samples=10, seed=1, classes=((2, 'Bg'), (3, 'Kratom')))
    cubes, _ = find_cubes(tmp_path)
    data = load_training_set(cubes)
    assert data.X.shape == (40 * 30 + 20 * 10, BANDS)
    assert data.category_names == {2: 'Bg', 3: 'Kratom', 1: 'Mango'} or data.category_names == \
        {1: 'Mango', 2: 'Bg', 3: 'Kratom'}
    assert sorted(set(data.y.tolist())) == [1, 2, 3]
    assert np.bincount(data.source).tolist() == [1200, 200]
    assert data.input_units == 'reflectance'
    assert np.allclose(data.wavelengths, WAVELENGTHS, atol=1e-3)


def test_training_on_a_folder_writes_one_model_for_all_classes(tmp_path):
    write_cube(tmp_path, 'one', seed=0, classes=((1, 'Mango'), (2, 'Bg')))
    write_cube(tmp_path, 'two', seed=1, classes=((2, 'Bg'), (3, 'Kratom')))
    out = tmp_path / 'out' / 'sgd.joblib'
    artifact = train_smart_bands_sgd(num_selected_bands=6, model_output=str(out), dataset_dir=str(tmp_path))
    saved = joblib.load(out)
    assert saved['category_names'] == {1: 'Mango', 2: 'Bg', 3: 'Kratom'}
    assert saved['classes'].tolist() == [1, 2, 3]
    assert [Path(p).name for p in saved['training_headers']] == ['one.hdr', 'two.hdr']
    assert Path(saved['training_header']).name == 'one.hdr'
    assert [Path(p).name for p in saved['training_labels']] == ['one.json', 'two.json']
    assert saved['W_fused'].shape == (6, 3) and saved['n_bands'] == BANDS
    assert saved['input_units'] == 'reflectance'
    assert artifact['classes'].tolist() == [1, 2, 3]


def test_one_cube_with_hdr_still_works(tmp_path):
    write_cube(tmp_path, 'solo', samples=30, classes=((1, 'Mango'), (2, 'Bg'), (3, 'Kratom')))
    write_cube(tmp_path, 'other', seed=3, classes=((1, 'Mango'), (2, 'Bg'), (3, 'Kratom')))
    out = tmp_path / 'solo.joblib'
    train_smart_bands_sgd(num_selected_bands=4, hdr_path=str(tmp_path / 'solo.hdr'), model_output=str(out),
                          dataset_dir=str(tmp_path))
    assert [Path(p).name for p in joblib.load(out)['training_headers']] == ['solo.hdr']    # not 'other'


def test_two_classes_are_refused_with_the_reason(tmp_path):
    write_cube(tmp_path, 'two_classes', classes=((1, 'Mango'), (2, 'Bg')))
    with pytest.raises(ValueError, match='at least 3 classes'):
        train_smart_bands_sgd(dataset_dir=str(tmp_path), model_output=str(tmp_path / 'm.joblib'))


def test_cubes_with_other_bands_are_refused(tmp_path):
    write_cube(tmp_path, 'one')
    write_cube(tmp_path, 'two', bands=BANDS + 2, wavelengths=np.linspace(900.0, 1700.0, BANDS + 2))
    with pytest.raises(ValueError, match='same bands'):
        load_training_set(find_cubes(tmp_path)[0])


def test_cubes_with_other_wavelengths_are_refused(tmp_path):
    write_cube(tmp_path, 'one')
    write_cube(tmp_path, 'two', wavelengths=WAVELENGTHS + 5.0)
    with pytest.raises(ValueError, match='same bands'):
        load_training_set(find_cubes(tmp_path)[0])


def test_labels_drawn_on_another_size_are_refused(tmp_path):
    write_cube(tmp_path, 'one', image_size=(99, 99))
    with pytest.raises(ValueError, match='labels are drawn on 99 x 99'):
        load_training_set(find_cubes(tmp_path)[0])


def test_an_empty_folder_says_what_is_missing(tmp_path):
    (tmp_path / 'only.hdr').write_text('ENVI')
    with pytest.raises(ValueError, match='no labelled cube'):
        train_smart_bands_sgd(dataset_dir=str(tmp_path))


def test_check_calibration_rebuilds_the_pooled_held_out_split(tmp_path, capsys):
    write_cube(tmp_path, 'one', seed=0, classes=((1, 'Mango'), (2, 'Bg')))
    write_cube(tmp_path, 'two', seed=1, classes=((2, 'Bg'), (3, 'Kratom')))
    out = tmp_path / 'sgd.joblib'
    train_smart_bands_sgd(num_selected_bands=6, model_output=str(out), dataset_dir=str(tmp_path))
    from tools import check_calibration
    check_calibration.main([str(out)])
    text = capsys.readouterr().out
    total = 40 * 30 + 40 * 30
    assert f"Pixels      {round(total * 0.2)} held-out 20 %" in text
    assert 'Accuracy' in text


def test_report_writes_every_figure_and_the_index(tmp_path):
    write_cube(tmp_path, 'one', seed=0, classes=((1, 'Mango'), (2, 'Bg'), (3, 'Kratom')))
    write_cube(tmp_path, 'two+x', seed=1, classes=((2, 'Bg'), (3, 'Kratom')))
    report = tmp_path / 'report'
    train_smart_bands_sgd(num_selected_bands=6, model_output=str(tmp_path / 'm.joblib'),
                          dataset_dir=str(tmp_path), report_dir=report)
    names = sorted(p.name for p in report.iterdir())
    assert names == ['calibration.png', 'index.html', 'map_one.png', 'map_two_x.png', 'overview.png',
                     'projection.png', 'report.txt', 'spectra.png', 'weights.png']
    text = (report / 'report.txt').read_text(encoding='utf-8')
    assert 'Confusion matrix' in text and 'Mango' in text and 'two+x.hdr' in text
    html = (report / 'index.html').read_text(encoding='utf-8')
    assert all(f"src='{n}'" in html for n in names if n.endswith('.png'))
