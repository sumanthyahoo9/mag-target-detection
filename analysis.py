"""
WHERE and WHY do the methods fail?
"""
from dataclasses import replace
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import HARD, Config
from evaluate import evaluate, find_peaks, match
from generate import make_dataset

HARD_KNOBS = ["target_burial_m", "target_remanence", "line_spacing_m", "noise_nT", "drift_nT"]


def recall_by_depth(hits: List[np.ndarray],
                    targets: List[np.ndarray],
                    edges: Tuple = (0, 1, 2, 3)) -> list[float]:
    """
    Recall at the operating threshold, per burial-depth band (column 2 of the target table).
    """
    h, d = np.concatenate(hits), np.concatenate([t[:, 2] for t in targets])
    return [float(h[(d >= lo) & (d < hi)].mean()) for lo, hi in zip(edges[:-1], edges[1:])]

def knob_sweep(score_fn: Callable[[np.ndarray], np.ndarray], 
               cfg: Config, n: int, seed: int) -> dict[str, float]:
    """
    Set ONE knob to its HARD value, keep the rest at default -> AP. 
    Isolates each cause of the HARD collapse.
    """
    out = {}
    for knob in HARD_KNOBS:
        c = replace(cfg, **{knob: getattr(HARD, knob)})
        x, _, t, _ = make_dataset(c, n, seed)
        out[knob] = evaluate(score_fn(x), t, c)["ap"]
    return out


def failure_gallery(maps: np.ndarray, scores: np.ndarray,
                    targets: list[np.ndarray], boulders: list[np.ndarray],
                    cfg: Config, path: str, thr: float = 0.5, n: int = 4) -> None:
    """
    First n patches with a miss or a false alarm. 
    Green x = found target, 
    Red x = missed target,
    White o = boulder centre, 
    Red square = false alarm (detection >= thr not within 1 m of a target).
    """
    L = cfg.patch_size_m
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    shown = 0
    for m, s, t, b in zip(maps, scores, targets, boulders):
        dets = find_peaks(s, cfg)
        dets = dets[dets[:, 2] >= thr]
        is_tp, hit = match(dets, t[:, :2])
        if hit.all() and is_tp.all():
            continue
        ax = axes[shown]
        v = np.percentile(np.abs(m - m.mean()), 99)
        ax.imshow(m - m.mean(), origin="lower", extent=[0, L, 0, L], cmap="RdBu_r", vmin=-v, vmax=v)
        ax.scatter(t[hit, 0], t[hit, 1], marker="x", c="lime", s=80)
        ax.scatter(t[~hit, 0], t[~hit, 1], marker="x", c="red", s=80)
        ax.scatter(b[:, 0], b[:, 1], marker="o", facecolors="none", edgecolors="w", s=120)
        ax.scatter(dets[~is_tp, 0], dets[~is_tp, 1], marker="s", facecolors="none", edgecolors="red", s=120)
        ax.set_title(" ".join(f"{z:.1f}m" for z in t[:, 2]) or "no targets", fontsize=9)
        shown += 1
        if shown == n:
            break
    fig.suptitle("Failures (title = burial depth of each target)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)