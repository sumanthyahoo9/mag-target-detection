"""
Rung 1 (no learning): amplitude and matched-filter detectors.
Rung 2 (simple ML: blob features + gradient boosting)
"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import fftconvolve
from sklearn.ensemble import HistGradientBoostingClassifier

from config import Config
from generate import dipole_anomaly, field_dir
from evaluate import find_peaks, match

# ---------------- Rung 1: amplitude and matched-filter detectors ----------------
def preprocess(maps: np.ndarray, cfg: Config, highpass_m: float = 5.0) -> np.ndarray:
    """
    Remove line-to-line drift (subtract each column's median: one column ~ one survey line)
    and broad geology (subtract a heavily blurred copy = high-pass filter). 
    Works on (H,W) or (N,H,W).
    """
    x = maps - np.median(maps, axis=-2, keepdims=True)
    s = highpass_m / cfg.res_m
    return x - gaussian_filter(x, sigma=(0, s, s)[-x.ndim:])


def target_template(cfg: Config, depth_m: float = 3.5, half_m: float = 7.5) -> np.ndarray:
    """
    Anomaly of an ideal, purely induced target `depth_m` below the sensor
    characteristics: zero-mean, unit-norm.
    """
    f_hat = field_dir(cfg)
    ax = np.arange(-half_m, half_m + cfg.res_m / 2, cfg.res_m)
    # rows = y, cols = x
    X, Y = np.meshgrid(ax, ax)
    obs = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    t = dipole_anomaly(obs, np.array([[0.0, 0.0, -depth_m]]), f_hat[None], f_hat).reshape(X.shape)
    t -= t.mean()
    return t / np.linalg.norm(t)


def amplitude_scores(maps: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Simplest detector: 
    Asks the question, HOW strong is the preprocessed field here? 
    (N,H,W) -> (N,H,W)
    """
    return np.abs(preprocess(maps, cfg))


def matched_filter_scores(maps: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Correlate each preprocessed map with the ideal-target template
    negatives clipped. 
    (N,H,W) -> (N,H,W)
    """
    # correlation = convolution with flipped template
    t = target_template(cfg)[::-1, ::-1]
    s = np.stack([fftconvolve(m, t, mode="same") for m in preprocess(maps, cfg)])
    return np.clip(s, 0, None)


def scale_to_unit(scores: np.ndarray, ref: np.ndarray, q: float = 99.9) -> np.ndarray:
    """
    Rescale raw scores to roughly [0,1] using a percentile of reference (validation) scores.
    Ranking metrics (AP, recall at a false-alarm budget) are unaffected 
    this only sets thresholds.
    """
    return scores / np.percentile(ref, q)

# ---------------- Rung 2: candidate blobs -> hand-crafted features -> gradient boosting ---------
FEATURES = ["mf_score", "high", "low", "low_high_ratio", "lobe_dist_m",
            "lobe_angle_cos", "lobe_angle_sin",
            "width_m", "n_lobes", "shape_corr", "local_std"]

def candidates(mf_scaled: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Matched-filter peaks at a low threshold (high recall). 
    (H,W) -> (K,3) rows of x, y, mf_score.
    """
    return find_peaks(mf_scaled, cfg, min_score=0.02)

def blob_features(resid: np.ndarray, cand: np.ndarray, cfg: Config,
                  half_m: float = 5.0) -> np.ndarray:
    """
    Describe the anomaly in a (2*half_m)-wide window around each candidate. 
    Returns (K, len(FEATURES)).
    """
    h = int(round(half_m / cfg.res_m))
    pad = np.pad(resid, h, mode="reflect")
    tmpl = target_template(cfg, half_m=half_m)
    rows = []
    for x, y, s in cand:
        r, c = int(round(y / cfg.res_m)), int(round(x / cfg.res_m))
        w = pad[r:r + 2 * h + 1, c:c + 2 * h + 1]                # window centred on the candidate
        (rh, ch), (rl, cl) = np.unravel_index(w.argmax(), w.shape), np.unravel_index(w.argmin(), w.shape)
        hi, lo = w.max(), w.min()
        dy, dx = (rl - rh) * cfg.res_m, (cl - ch) * cfg.res_m  # vector from high lobe to low lobe
        dist = np.hypot(dx, dy) + 1e-6
        a = np.abs(w)
        lobes = (a == maximum_filter(a, size=5)) & (a > 0.3 * a.max())
        wz = w - w.mean()
        rows.append([s, hi, lo, -lo / (hi + 1e-6), dist, dx / dist, dy / dist,
                     np.sqrt((a > 0.5 * a.max()).sum()) * cfg.res_m, lobes.sum(),
                     (wz * tmpl).sum() / (np.linalg.norm(wz) + 1e-6), w.std()])
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURES))

def candidate_table(maps: np.ndarray, cfg: Config, mf_ref: np.ndarray
                    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Returns, per patch: 
    candidate positions (K,3)
    features (K,F).
    """
    resid, mf = preprocess(maps, cfg), scale_to_unit(matched_filter_scores(maps, cfg), mf_ref)
    cands = [candidates(m, cfg) for m in mf]
    return cands, [blob_features(r, c, cfg) for r, c in zip(resid, cands)]


def fit_blob_classifier(maps: np.ndarray, targets: list[np.ndarray], cfg: Config, mf_ref: np.ndarray
                        ) -> HistGradientBoostingClassifier:
    """
    Label each candidate 
    1 if within 1 m of a true target (same rule as evaluation), 
    else 0 
    Fit a Gradient Boosting Machine.
    """
    cands, feats = candidate_table(maps, cfg, mf_ref)
    y = np.concatenate([match(c, t[:, :2])[0] for c, t in zip(cands, targets)])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    return clf.fit(np.concatenate(feats), y)

def blob_scores(clf: HistGradientBoostingClassifier, 
                maps: np.ndarray, cfg: Config, mf_ref: np.ndarray
                ) -> np.ndarray:
    """
    Turn classifier probabilities back into sparse score maps so the shared evaluate() applies.
    """
    out = np.zeros(maps.shape, dtype=np.float32)
    for i, (c, f) in enumerate(zip(*candidate_table(maps, cfg, mf_ref))):
        if len(c):
            rr, cc = np.round(
                c[:, 1] / cfg.res_m).astype(int), np.round(c[:, 0] / cfg.res_m).astype(int)
            out[i, rr, cc] = clf.predict_proba(f)[:, 1]
    return out
