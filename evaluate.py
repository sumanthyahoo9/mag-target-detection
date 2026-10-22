"""
Shared evaluation: every method outputs a score map, and is scored by the same code.
Steps:
1. score map
2. find_peaks
3. (x, y, confidence) detections
4. match to ground truth within radius_m
5. precision-recall curve, average precision, and an operating point.
"""
from typing import Tuple
import numpy as np
from scipy.ndimage import maximum_filter
from config import Config


def find_peaks(score: np.ndarray, cfg: Config, 
               min_score: float = 0.05, min_dist_m: float = 1.5) -> np.ndarray:
    """
    Steps:
    Local maxima of a score map
    (K,3) array of (x_m, y_m, confidence), highest first.
    """
    size = 2 * int(round(min_dist_m / cfg.res_m)) + 1
    is_peak = (score == maximum_filter(score, size=size, mode="nearest")) & (score >= min_score)
    rows, cols = np.nonzero(is_peak)
    dets = np.c_[cols * cfg.res_m, rows * cfg.res_m, score[rows, cols]]
    return dets[np.argsort(-dets[:, 2])]


def match(dets: np.ndarray, gts: np.ndarray, 
          radius_m: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy matching, highest confidence first; each ground truth can be matched once.
    Returns (is_tp per detection, hit per ground-truth target).
    """
    is_tp = np.zeros(len(dets), bool)
    hit = np.zeros(len(gts), bool)
    for i, (x, y, _) in enumerate(dets):
        if len(gts) == 0:
            break
        d = np.hypot(gts[:, 0] - x, gts[:, 1] - y)
        d[hit] = np.inf
        j = np.argmin(d)
        if d[j] <= radius_m:
            is_tp[i] = hit[j] = True
    return is_tp, hit


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Area under the precision-recall curve (precision made monotonic, as in VOC/COCO).
    """
    if len(recall) == 0:
        return 0.0
    p = np.maximum.accumulate(np.r_[precision, 0.0][::-1])[::-1]
    r = np.r_[0.0, recall]
    return float(np.sum((r[1:] - r[:-1]) * p[:-1]))


def evaluate(scores: np.ndarray, targets: list[np.ndarray], 
             cfg: Config, radius_m: float = 1.0, 
             op_threshold: float = 0.5, fa_budget: float = 0.5) -> dict:
    """
    (Score maps (N,H,W) + ground-truth target lists) --> metrics dict.
    ap            : average precision over all confidence thresholds
    recall_at_fa  : best recall with at most `fa_budget` false alarms per patch
    recall_op, fa_per_patch_op : recall and false alarms per patch at `op_threshold`
    hits_op       : per-patch boolean arrays (was each target found?) for depth/strength analysis
    """
    conf, tp, hits_op, fa_op, n_gt = [], [], [], 0, 0
    for s, gt in zip(scores, targets):
        gt_xy = gt[:, :2]
        dets = find_peaks(s, cfg)
        conf.append(dets[:, 2])
        tp.append(match(dets, gt_xy, radius_m)[0])
        is_tp_op, hit = match(dets[dets[:, 2] >= op_threshold], gt_xy, radius_m)
        hits_op.append(hit)
        fa_op += int((~is_tp_op).sum())
        n_gt += len(gt)

    conf, tp = np.concatenate(conf), np.concatenate(tp)
    order = np.argsort(-conf)
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    recall = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1)
    fa_per_patch = cfp / len(scores)
    ok = fa_per_patch <= fa_budget
    return dict(
        ap=average_precision(precision, recall),
        precision=precision, recall=recall, thresholds=conf[order],
        recall_at_fa=float(recall[ok].max()) if ok.any() else 0.0,
        recall_op=float(np.concatenate(hits_op).mean()) if n_gt else float("nan"),
        fa_per_patch_op=fa_op / len(scores),
        hits_op=hits_op,
    )
