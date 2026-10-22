"""
Unit tests: physics, labels, and the shared evaluation. 
Run: python -m pytest -q
"""
import numpy as np

from config import Config
from evaluate import evaluate, find_peaks, match
from generate import dipole_anomaly, field_dir, gaussian_label, make_dataset

CFG = Config()


def test_dipole_falls_off_as_inverse_cube() -> None:
    """
    Physics sanity check, inverse-cubed law
    """
    f_hat = field_dir(CFG)
    y = np.linspace(-15, 15, 601)
    obs = np.c_[np.zeros_like(y), y, np.zeros_like(y)]
    peak = lambda d: np.abs(dipole_anomaly(obs, np.array([[0, 0, -d]]), f_hat[None] * 2.0, f_hat)).max()
    assert abs(peak(2.0) / peak(4.0) - 8.0) < 0.1


def test_label_peaks_exactly_at_target() -> None:
    """
    Test whether the label peaks exactly at the target
    """
    lab = gaussian_label(np.array([[10.0, 20.0]]), CFG)          # x=10 m -> col 20, y=20 m -> row 40
    assert np.unravel_index(lab.argmax(), lab.shape) == (40, 20)
    assert np.isclose(lab.max(), 1.0) and lab.min() >= 0.0


def test_dataset_shapes() -> None:
    """
    Shape sanity check
    We want the shapes to be uniform squares, ideally
    """
    maps, labels, targets, boulders = make_dataset(CFG, 4, seed=0)
    assert maps.shape == labels.shape == (4, CFG.grid_n, CFG.grid_n)
    assert all(t.shape[1] == 4 for t in targets) and all(b.shape[1] == 2 for b in boulders)


def test_peaks_and_matching_on_toy_case() -> None:
    """
    Check whether the peaks match with the labels/matching
    """
    gts = np.array([[10.0, 10.0]])
    score = gaussian_label(np.array([[10.0, 10.5], [30.0, 30.0]]), CFG)   # one near-hit, one false alarm
    dets = find_peaks(score, CFG)
    assert len(dets) == 2
    is_tp, hit = match(dets, gts, radius_m=1.0)
    assert is_tp.sum() == 1 and hit.all()


def test_empty_patch_gives_only_false_alarms() -> None:
    """
    False alarms check
    """
    dets = np.array([[5.0, 5.0, 0.9]])
    is_tp, hit = match(dets, np.zeros((0, 2)))
    assert not is_tp.any() and len(hit) == 0


def test_perfect_scores_give_near_perfect_ap() -> None:
    """
    Check whether perfect scores give near perfect precision
    """
    _, labels, targets, _ = make_dataset(CFG, 50, seed=1)
    res = evaluate(labels, targets, CFG)            # the label map itself = an oracle predictor
    assert res["ap"] > 0.95 and res["fa_per_patch_op"] == 0.0