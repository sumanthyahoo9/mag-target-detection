"""
Physics-based synthetic magnetometer survey patches.

Coordinates: x = east, y = north, z = up (metres). Maps are indexed [row = y, col = x],
so row 0 is the southern edge: plot with origin="lower" to put north at the top.
"""
from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d

from config import Config

MU0_4PI_NT = 100.0  # mu0/4pi = 1e-7 T·m/A  ->  100 nT·m^3 per A·m^2


def field_dir(cfg: Config) -> np.ndarray:
    """Unit vector of Earth's main field (x east, y north, z up)."""
    inc, dec = np.radians(cfg.inclination_deg), np.radians(cfg.declination_deg)
    return np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])


def dipole_anomaly(obs: np.ndarray, src: np.ndarray, moment: np.ndarray, f_hat: np.ndarray) -> np.ndarray:
    """Total-field anomaly (nT) at observation points from a set of point dipoles.

    obs: (P,3) sensor positions; src: (S,3) dipole positions; moment: (S,3) in A·m^2.
    The total-field anomaly is the dipole field projected onto Earth's field direction
    (valid because anomalies are tiny compared with the ~50,000 nT main field).
    """
    r = obs[:, None, :] - src[None, :, :]                   # (P,S,3)
    dist = np.linalg.norm(r, axis=-1, keepdims=True)        # (P,S,1)
    r_hat = r / dist
    m_dot_r = np.sum(moment[None] * r_hat, axis=-1, keepdims=True)
    b = MU0_4PI_NT * (3 * m_dot_r * r_hat - moment[None]) / dist**3   # (P,S,3) nT
    return b.sum(axis=1) @ f_hat                            # (P,)


def random_unit(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Normalize
    """
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def gaussian_label(xy: np.ndarray, cfg: Config) -> np.ndarray:
    """Soft label map: a Gaussian bump at each target centre (max where bumps overlap)."""
    n = cfg.grid_n
    rows, cols = np.mgrid[0:n, 0:n]
    lab = np.zeros((n, n), np.float32)
    for x, y in xy:
        c, r = x / cfg.res_m, y / cfg.res_m
        lab = np.maximum(lab, np.exp(-((cols - c) ** 2 + (rows - r) ** 2) / (2 * cfg.label_sigma_px ** 2)))
    return lab


def make_patch(cfg: Config, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, 
                                                               np.ndarray, np.ndarray]:
    """One survey patch -> (map [N,N] nT, label [N,N], targets [T,4], boulder centres [B,2])."""
    L, n = cfg.patch_size_m, cfg.grid_n
    f_hat = field_dir(cfg)
    lo, hi = cfg.margin_m, L - cfg.margin_m

    # Targets: one compact dipole each; magnetisation mostly induced (along Earth's field)
    # plus a remanent part in a random direction, blended by `target_remanence`
    nt = rng.integers(cfg.n_targets[0], cfg.n_targets[1] + 1)
    t_xy = rng.uniform(lo, hi, (nt, 2))
    t_burial = rng.uniform(*cfg.target_burial_m, nt)
    t_mom = rng.uniform(*cfg.target_moment, nt)
    q = cfg.target_remanence
    t_dir = (1 - q) * f_hat + q * random_unit(rng, nt)
    t_dir /= np.linalg.norm(t_dir, axis=1, keepdims=True)
    src, mom = [np.c_[t_xy, -t_burial]], [t_dir * t_mom[:, None]]

    # Boulders: clusters of 2-4 small sources, each magnetised in a random direction
    nb = rng.integers(cfg.n_boulders[0], cfg.n_boulders[1] + 1)
    b_xy = rng.uniform(lo, hi, (nb, 2))
    for centre in b_xy:
        k = rng.integers(cfg.boulder_sources[0], cfg.boulder_sources[1] + 1)
        offsets = rng.uniform(-cfg.boulder_radius_m, cfg.boulder_radius_m, (k, 2))
        src.append(np.c_[centre + offsets, -rng.uniform(*cfg.boulder_burial_m, k)])
        mom.append(random_unit(rng, k) * rng.uniform(*cfg.boulder_moment, k)[:, None])
    src, mom = np.vstack(src), np.vstack(mom)

    # Survey: north-south lines, sampled at the grid resolution along each line
    x_lines = np.arange(0, L + cfg.line_spacing_m, cfg.line_spacing_m)
    X, Y = np.meshgrid(x_lines, np.arange(n) * cfg.res_m)          # (n_rows, n_lines)
    obs = np.c_[X.ravel(), Y.ravel(), np.full(X.size, cfg.sensor_alt_m)]
    vals = (dipole_anomaly(obs, src, mom, f_hat).reshape(X.shape)
            if len(src) else np.zeros(X.shape))

    # Geology: broad, smooth variation from a few long-wavelength waves (15-60 m)
    for _ in range(4):
        k = 2 * np.pi / rng.uniform(15, 60)
        ang, phase = rng.uniform(0, 2 * np.pi, 2)
        vals += cfg.geology_nT / 4 * np.cos(k * (X * np.cos(ang) + Y * np.sin(ang)) + phase)

    # Residual daily drift: lines are flown one after another, so the offset wanders
    # line to line (a random walk) -> stripes along the north-south lines
    drift = np.cumsum(rng.normal(0, cfg.drift_nT, len(x_lines)))
    vals += drift - drift.mean()
    vals += rng.normal(0, cfg.noise_nT, vals.shape)                # sensor noise

    # Grid: interpolate across lines onto the regular x grid (between lines = estimated, not measured)
    x_grid = np.arange(n) * cfg.res_m
    grid = interp1d(x_lines, vals, axis=1)(x_grid).astype(np.float32)

    return grid, gaussian_label(t_xy, cfg), np.c_[t_xy, t_burial, t_mom], b_xy


def make_dataset(cfg: Config, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, 
                                                          list[np.ndarray], 
                                                          list[np.ndarray]]:
    """n independent patches from one seed -> (maps, labels, targets list, boulders list)."""
    rng = np.random.default_rng(seed)
    out = [make_patch(cfg, rng) for _ in range(n)]
    maps, labels, targets, boulders = zip(*out)
    return np.stack(maps), np.stack(labels), list(targets), list(boulders)