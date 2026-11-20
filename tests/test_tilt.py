import json

import mrcfile
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, minimum_filter

from dogpick.cli import main
from dogpick.tilt import TiltParams, pick_image


def _blobs(shape=(200, 240), centers=None, d=16.0, amp=1.0, noise=0.05, seed=0):
    """Dark Gaussian particles at (x, y) centres; returns image and centres."""
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = [(40.0 + 50 * i + 0.3 * i, 45.0 + 55 * j + 0.2 * j) for i in range(4) for j in range(3)]
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    img = np.zeros(shape)
    for x, y in centers:
        img -= amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (d / 4) ** 2))
    return img + rng.normal(0, noise, shape), np.asarray(centers)


def _reference_research_dog(img, d, n):
    """The benchmarked research recipe (dog_screen.py dog_pick / export_jolt_picks.py), verbatim."""
    img = (img - img.mean()) / (img.std() + 1e-12)
    s1 = d / (2 * np.sqrt(2))
    dog = gaussian_filter(img, s1) - gaussian_filter(img, 1.6 * s1)
    nms = max(1, int(round(d / 2)))
    m = (dog == minimum_filter(dog, size=2 * nms + 1)) & (dog < 0)
    m[:nms, :] = m[-nms:, :] = m[:, :nms] = m[:, -nms:] = False
    iy, ix = np.nonzero(m)
    o = np.argsort(dog[iy, ix], kind="stable")[:n]
    return np.column_stack([ix[o], iy[o]])


def _nearest(picks, c):
    return np.min(np.hypot(picks[:, 0] - c[0], picks[:, 1] - c[1]))


def test_window_frac_1_reproduces_research_recipe_exactly():
    img, _ = _blobs(noise=0.3, seed=3)
    ref = _reference_research_dog(img, 16.0, 50)
    got = pick_image(img, TiltParams(diameter_px=16.0, top_k=50, window_frac=1.0))
    assert np.array_equal(got[:, :2].astype(int), ref)


def test_recovers_dark_particles_and_ignores_bright():
    img, centers = _blobs()
    pk = pick_image(img, TiltParams(diameter_px=16.0, top_k=None))
    assert all(_nearest(pk, c) <= 1.0 for c in centers)
    bright = pick_image(-img, TiltParams(diameter_px=16.0, top_k=None))
    assert all(_nearest(bright, c) > 4 for c in centers) if len(bright) else True
    pk_b = pick_image(-img, TiltParams(diameter_px=16.0, top_k=None, bright=True))
    assert all(_nearest(pk_b, c) <= 1.0 for c in centers)


def test_top_k_and_score_order():
    img, _ = _blobs(noise=0.5, seed=1)
    pk = pick_image(img, TiltParams(diameter_px=16.0, top_k=7))
    assert len(pk) == 7 and np.all(np.diff(pk[:, 2]) <= 0)
    allc = pick_image(img, TiltParams(diameter_px=16.0, top_k=None))
    assert np.array_equal(pk, allc[:7])


def test_smaller_window_and_positive_minima_only_add_candidates():
    img, _ = _blobs(noise=0.5, seed=2)
    n1 = len(pick_image(img, TiltParams(diameter_px=16.0, top_k=None, window_frac=1.0)))
    n05 = len(pick_image(img, TiltParams(diameter_px=16.0, top_k=None, window_frac=0.5)))
    npos = len(pick_image(img, TiltParams(diameter_px=16.0, top_k=None, require_negative=False)))
    assert n05 >= n1 and npos >= n05


def test_subpixel_off_is_integer_and_on_is_more_accurate():
    rng = np.random.default_rng(7)
    centers = [(30 + 45 * i + rng.uniform(-0.5, 0.5), 30 + 45 * j + rng.uniform(-0.5, 0.5))
               for i in range(5) for j in range(4)]
    img, c = _blobs(shape=(220, 260), centers=centers, noise=0.0)
    off = pick_image(img, TiltParams(diameter_px=16.0, top_k=None))
    on = pick_image(img, TiltParams(diameter_px=16.0, top_k=None, subpixel=True))
    assert np.all(off[:, :2] == np.round(off[:, :2]))
    err = lambda p: np.mean([_nearest(p, cc) for cc in c])
    assert err(on) < err(off) and err(on) < 0.1


def test_scores_comparable_across_contrast():
    """Robust per-image z-score: scaling the image does not change scores."""
    img, _ = _blobs(noise=0.3, seed=4)
    a = pick_image(img, TiltParams(diameter_px=16.0, top_k=10))
    b = pick_image(5.0 * img + 3.0, TiltParams(diameter_px=16.0, top_k=10))
    assert np.allclose(a, b)


def test_params_validation():
    with pytest.raises(ValueError):
        TiltParams(diameter_px=0)
    with pytest.raises(ValueError):
        TiltParams(diameter_px=10, top_k=0)
    with pytest.raises(ValueError):
        pick_image(np.zeros((5, 10, 10)), TiltParams(diameter_px=4))


# ---- CLI ------------------------------------------------------------------------------------------
def _write_mrc(path, data, apix=10.0):
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.asarray(data, np.float32))
        m.voxel_size = apix


def _star_rows(path):
    return [ln.split("\t") for ln in path.read_text().splitlines() if ln.startswith(("tilt_", "TS_"))]


def test_cli_stack_layout_matches_jolt_detect_2d(tmp_path):
    img, centers = _blobs()
    st = tmp_path / "TS_7.st"
    _write_mrc(st, np.stack([img, img, img]))
    assert main(["tilt", str(st), "--diameter", "160", "--out", str(tmp_path / "picks")]) == 0
    out = tmp_path / "picks" / "TS_7" / "TS_7_picks.star"
    text = out.read_text().splitlines()
    assert text[:3] == ["", "data_", ""] and text[3] == "loop_"
    assert text[4:9] == ["_rlnMicrographName #1", "_rlnCoordinateX #2", "_rlnCoordinateY #3",
                         "_rlnAutopickFigureOfMerit #4", "_TomoName #5"]
    rows = _star_rows(out)
    assert {r[0] for r in rows} == {"tilt_01", "tilt_02", "tilt_03"} and {r[4] for r in rows} == {"TS_7"}
    assert all(r[1].lstrip("-").isdigit() for r in rows)  # integer px by default
    meta = json.loads((out.parent / "TS_7_picks.star.json").read_text())
    assert meta["params"]["diameter_px"] == pytest.approx(16.0) and meta["params"]["top_k"] == 1200
    assert meta["params"]["nms_window_px"] == 9 and meta["n_tilts"] == 3


def test_cli_images_one_star(tmp_path):
    img, _ = _blobs()
    files = []
    for k in (1, 2):
        f = tmp_path / f"TS_7_{k:02d}.mrc"
        _write_mrc(f, img)
        files.append(str(f))
    star = tmp_path / "out" / "TS_7_picks.star"
    assert main(["tilt", *files, "--diameter_px", "16", "--out_star", str(star), "--ts_name", "TS_7",
                 "--subpixel", "--top_k", "5"]) == 0
    rows = _star_rows(star)
    assert {r[0] for r in rows} == {"TS_7_01.mrc", "TS_7_02.mrc"} and len(rows) == 10
    assert all("." in r[1] for r in rows)  # sub-pixel floats


def test_cli_requires_pixel_size_for_angstrom(tmp_path):
    img, _ = _blobs()
    st = tmp_path / "TS_8.st"
    _write_mrc(st, np.stack([img, img]), apix=0.0)
    with pytest.raises(SystemExit):
        main(["tilt", str(st), "--diameter", "160", "--out", str(tmp_path / "p")])
    assert main(["tilt", str(st), "--diameter", "160", "--apix", "10", "--out", str(tmp_path / "p")]) == 0


def test_output_reads_in_jolt(tmp_path):
    """JOLT's own STAR reader and tilt-index parser accept the output (skipped if jolt is not installed)."""
    star_mod = pytest.importorskip("jolt.io.star")
    pipeline = pytest.importorskip("jolt.pipeline")
    img, _ = _blobs()
    st = tmp_path / "TS_9.st"
    _write_mrc(st, np.stack([img] * 12))
    main(["tilt", str(st), "--diameter", "160", "--out", str(tmp_path / "p")])
    table = star_mod.read_star_table(tmp_path / "p" / "TS_9" / "TS_9_picks.star",
                                     required=["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"])
    idx = {pipeline.tilt_index_from_name(n) for n in table["rlnMicrographName"].astype(str)}
    assert idx == set(range(1, 13))
    assert table["rlnAutopickFigureOfMerit"].astype(float).notna().all()
