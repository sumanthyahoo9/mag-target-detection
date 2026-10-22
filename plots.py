"""
Visual and physics checks of the synthetic data. Run: python plots.py  -> figures in outputs/
"""
import os
from typing import List
from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import numpy as np

from config import Config
from generate import dipole_anomaly, field_dir, make_dataset

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


def extent(cfg: Config) -> List[float]:
    """
    Pixel centres sit at the centre
    """
    h = cfg.res_m / 2                       # pixel centres sit at 0, 0.5, ... 49.5 m
    return [-h, cfg.patch_size_m - h, -h, cfg.patch_size_m - h]


def show_map(ax: Axes, m: np.ndarray, cfg: Config, targets: np.ndarray | tuple = (), 
             boulders: np.ndarray | tuple = (), title: str = "", vlim: float | None = None) -> AxesImage:
    """
    Heat map of the targets and boulders
    """
    v = vlim or np.percentile(np.abs(m), 99)
    im = ax.imshow(m, origin="lower", extent=extent(cfg), cmap="RdBu_r", vmin=-v, vmax=v)
    if len(targets) > 0:
        ax.scatter(targets[:, 0], targets[:, 1], marker="x", c="k", s=60, label="target")
    if len(boulders) > 0:
        ax.scatter(boulders[:, 0], boulders[:, 1], marker="o", facecolors="none",
                   edgecolors="lime", s=90, label="boulder")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="upper right")
    ax.set(title=title, xlabel="x east (m)", ylabel="y north (m)")
    return im


def peak_near(m: np.ndarray, xy: np.ndarray, 
              cfg: Config, radius_m: float = 2.0) -> List[float]:
    """
    Largest |anomaly| within radius_m of each (x, y): a rough per-object strength.
    """
    yy, xx = np.mgrid[0:cfg.grid_n, 0:cfg.grid_n] * cfg.res_m
    return [np.abs(m[(xx - x) ** 2 + (yy - y) ** 2 <= radius_m ** 2]).max() for x, y in xy]


def physics_check(cfg: Config) -> None:
    """
    One induced target: north-south profiles at several depths, 
    Check whether peak vs distance conforms to 1/r**3.
    """
    f_hat = field_dir(cfg)
    y = np.linspace(-15, 15, 301)
    obs = np.c_[np.zeros_like(y), y, np.zeros_like(y)]        # sensor line passing over the object
    dists = np.array([1.0, 2, 3, 4, 6, 8])
    peaks = []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for d in dists:
        prof = dipole_anomaly(obs, np.array([[0.0, 0.0, -d]]), f_hat[None] * 2.0, f_hat)
        peaks.append(np.abs(prof).max())
        if d in (2, 4, 6):
            a1.plot(y, prof, label=f"{d:.0f} m below sensor")
    a1.axvline(0, c="grey", ls=":", label="object position")
    a1.set(title="North-south profile over one induced target (2 A·m²)",
           xlabel="y north (m)", ylabel="anomaly (nT)")
    a1.legend(fontsize=8)
    a2.loglog(dists, peaks, "o-", label="dipole peak")
    a2.loglog(dists, peaks[0] * (dists / dists[0]) ** -3, "--", label="1/r³ reference")
    a2.set(title="Peak |anomaly| vs distance", xlabel="distance (m)", ylabel="peak (nT)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/3_physics_check.png", dpi=120)
    print(f"Peak at 2 m / peak at 4 m = {peaks[1] / peaks[3]:.1f}x (expect 8x)")


if __name__ == "__main__":
    cfg = Config()
    clean_cfg = replace(cfg, geology_nT=0.0, drift_nT=0.0, noise_nT=0.0)

    # 1. Gallery: what the model sees, with ground truth overlaid
    maps, labels, T, B = make_dataset(cfg, 6, seed=42)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, ax in enumerate(axes.flat):
        im = show_map(ax, maps[i], cfg, T[i], B[i], f"Patch {i}: {len(T[i])} targets, {len(B[i])} boulders")
        fig.colorbar(im, ax=ax, label="nT", shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/1_gallery.png", dpi=120)

    # 2. Same patch, peeled apart: objects only vs full map vs label (same seed -> same objects)
    clean, _, _, _ = make_dataset(clean_cfg, 6, seed=42)
    i = 2
    v = np.percentile(np.abs(maps[i]), 99)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for a, m, t in zip(ax[:2], [clean[i], maps[i]], ["Objects only", "Full map (+ geology, drift, noise)"]):
        fig.colorbar(show_map(a, m, cfg, T[i], B[i], t, vlim=v), ax=a, label="nT")
    lab = ax[2].imshow(labels[i], origin="lower", extent=extent(cfg), cmap="viridis", vmin=0, vmax=1)
    ax[2].set(title="Label: Gaussian bumps at targets only", xlabel="x east (m)", ylabel="y north (m)")
    fig.colorbar(lab, ax=ax[2])
    fig.tight_layout()
    fig.savefig(f"{OUT}/2_decomposition.png", dpi=120)

    # 3. Physics sanity check
    physics_check(cfg)

    # 4. Can amplitude alone separate targets from boulders? (objects-only maps)
    cm, _, ct, cb = make_dataset(clean_cfg, 300, seed=7)
    tp = [p for m, t in zip(cm, ct) for p in peak_near(m, t[:, :2], cfg)]
    bp = [p for m, b in zip(cm, cb) for p in peak_near(m, b, cfg)]
    plt.figure(figsize=(7, 4))
    bins = np.logspace(np.log10(0.5), np.log10(max(tp + bp)), 40)
    plt.hist(tp, bins=bins, alpha=0.6, label=f"targets (n={len(tp)})")
    plt.hist(bp, bins=bins, alpha=0.6, label=f"boulders (n={len(bp)})")
    plt.xscale("log")
    plt.xlabel("peak |anomaly| near object (nT)")
    plt.ylabel("count")
    plt.title("Amplitude overlap: strength alone can't separate the classes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/4_amplitude_overlap.png", dpi=120)
    print(f"Median peak: targets {np.median(tp):.1f} nT | boulders {np.median(bp):.1f} nT")
