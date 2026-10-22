# Detecting buried metal objects in synthetic magnetometer surveys

A magnetometer towed above the seabed records tiny distortions (a few nT on top of Earth's ~50,000 nT field) caused by buried ferrous objects: anchors, chains, debris, unexploded ordnance. The task is to turn a gridded survey map into a **ranked list of target positions** for a diver or ROV to verify. The hard part is that **magnetic boulders, geology, survey drift and noise** produce anomalies of similar strength.

## Why this topic
I had not worked with magnetometer data before and wanted a problem where **physics shapes the data**, so understanding the signal matters as much as the model. It also has properties that make a good synthetic challenge:
- Ground truth is cheap to simulate exactly (point-dipole physics) but expensive to obtain in reality (each verification is a dive).
- It has **real structure**: confounders that look like targets, and a signal that fades as 1/r³.
- It has a natural **difficulty dial** (depth, remanence, line spacing, noise) for honest evaluation.
- The output is operationally meaningful: precision costs dive time, recall costs missed hazards.

## The data (`generate.py`, `config.py`)
Each sample is a **50 m × 50 m patch**, gridded at 0.5 m → a **100 × 100 map** of the residual field (nT).

| Component | Model | Why it makes the problem interesting |
|---|---|---|
| **Targets** (0–3 per patch) | Single point dipole, 0–3 m burial. Magnetisation = mix of induced (along Earth's field) and remanent (random direction), set by `target_remanence` | Faint when deep; pattern rotates with remanence; the peak is **offset** from the true position |
| **Boulders** (0–4 per patch) | Clusters of 2–4 randomly oriented dipoles | Same strength as targets (rebalanced so amplitude alone can't separate them); differ only in **shape** |
| **Geology** | Sum of long-wavelength waves (15–60 m) | Broad background of 10–20 nT |
| **Drift** | Random-walk offset per survey line | Stripes along the survey direction |
| **Survey + noise** | North–south lines every 2 m, sensor 2 m above the seabed, Gaussian noise, interpolated to the grid | Targets between lines are smeared |

Physics is checked in `plots.py`: the 1/r³ decay (8.0× from 2 m to 4 m), the south-high/north-low polarity for inclination 65°, and the southward peak offset.

**Labels:** a Gaussian heatmap (σ = 1 m) centred on each **true** target position; boulders are 0. The task is **keypoint detection**: map in → score map out → local peaks → (x, y, confidence).

All knobs live in `config.py`. `HARD` is a stress-test configuration: deeper, more remanent targets, wider line spacing, more noise and drift.

## Methods: a ladder from no learning to deep learning
| Rung | Method | Idea |
|---|---|---|
| 1a | Amplitude | Remove drift and geology (column median + high-pass), score = \|field\| |
| 1b | Matched filter | Correlate with the anomaly of an ideal induced target |
| 2 | Blob features + gradient boosting | Matched-filter peaks become candidates; 11 hand-crafted shape features per candidate; GBM classifies target vs not-target |
| 3 | Small U-Net (~100k params) | Learns the score map directly; CenterNet-style focal loss; best-validation-AP checkpoint |

Each rung must justify the next: the U-Net is only worth its complexity if it clearly beats the GBM.

## Evaluation (`evaluate.py`)
A detection is correct if it lies **within 1 m** of an unmatched target (greedy matching, highest confidence first).

- **AP:** average precision over all thresholds.
- **Recall @ ≤0.5 FA/patch:** the best recall achievable with at most one false alarm per two patches, a realistic dive budget.
- **Operating point (threshold 0.5):** recall and false alarms per patch.

The pipeline was sanity-checked: perfect scores give AP 0.998 and random scores give AP 0.0.

Splits are independently generated (train 2000 / val 300 / test 500 patches; HARD 500). Thresholds and checkpoints use **validation only**.

## Results
| Method | AP (test) | Recall @ ≤0.5 FA/patch | Recall / FA per patch @ 0.5 | AP (HARD) |
|---|---|---|---|---|
| Amplitude | 0.17 | 0.18 | 0.31 / 2.63 | 0.003 |
| Matched filter | 0.38 | 0.35 | 0.41 / 0.96 | 0.002 |
| Blob GBM | 0.51 | 0.55 | 0.43 / 0.15 | 0.005 |
| **U-Net** | **0.85** | **0.87** | **0.86 / 0.41** | 0.029 |

- **No learning fails.** Amplitude is weak because boulders are just as strong; a single fixed template can't cover every depth and magnetisation direction.
- **The GBM re-ranks, it doesn't find.** On the same candidates, its ranking AP is 0.84 vs 0.60 for the raw matched-filter score. Almost all the gain comes from one feature, the amplitude-normalised template correlation ("is the shape target-like, regardless of strength?"). But only **64%** of targets ever become candidates, which caps its recall.
- **The U-Net removes that ceiling** by learning its own "templates" for all depths and orientations and outputting positions directly.

## Where it fails (`analysis.py`, `outputs/5_failures.png`)
**Recall at threshold 0.5 by burial depth:**

| Method | 0–1 m | 1–2 m | 2–3 m |
|---|---|---|---|
| Blob GBM | 0.66 | 0.44 | 0.19 |
| U-Net | 0.94 | 0.94 | 0.71 |

**U-Net AP with one knob set to its HARD value** (default 0.85):

| Knob | AP |
|---|---|
| remanence 0.3 → 0.8 | **0.15** |
| burial → 1–4 m | 0.69 |
| line spacing 2 → 3 m | 0.71 |
| noise 0.5 → 1 nT | 0.74 |
| drift 3 → 5 nT | 0.85 |

- **Remanence dominates.** The U-Net learned the near-induced pattern it was trained on; strongly rotated patterns are out of distribution. This is a training-data problem, not a physics one.
- **Depth, line spacing and noise** are mostly physical limits. On HARD, 26% of targets peak below 2× the noise level even before geology and boulders are added, so no method can find them.
- **Drift is solved** by per-line median removal.
- **Typical errors in the gallery:** boulders that happen to look like a single clean dipole (false alarms); targets merged with a nearby boulder cluster, or deep and faint (misses).

**HARD** (last column of the Results table) sets **all five knobs above to their HARD values at once**. Every method scores AP ≤ 0.03 there (U-Net 0.029, others ≤ 0.005), because the effects compound and the data lies outside the training distribution. It is reported as a stress test, not as a target.

## Limitations
- **Synthetic physics:** point dipoles only (no extended objects such as pipelines or chains), flat seabed, constant sensor altitude, white noise, no heading error or tow-fish motion. Non-ferrous metals are invisible to magnetometers and are out of scope.
- **Noise on the numbers:** single training seed; run-to-run differences of ~0.02 AP between CPU and Apple-GPU runs should be read as noise.
- **No early stopping:** all epochs run and the best validation checkpoint is kept. Validation AP plateaus after ~6–8 epochs (mild overfitting).
- **Fixed thresholds:** the operating point is fixed at 0.5 and not calibrated per survey condition.

## Future work
1. **`--train-hard` experiment (implemented, not run for this submission):** retrain the U-Net on HARD data to separate the physical ceiling from distribution shift.
2. **Train with remanence spanning 0–1** (the sweep's main failure), and with a wider depth range.
3. Pair the knob sweep with the test seed and size; average results over several training seeds.
4. Patience-based early stopping; probability calibration for the operating threshold.
5. Use survey geometry explicitly (along-line sampling, sensor altitude) rather than interpolated grids.

## Running it
Tested with **Python 3.11.11** (needs 3.10+ for `X | None` type hints).
```bash
git clone https://github.com/sumanthyahoo9/mag-target-detection.git
cd mag-target-detection
python -m venv .venv && source .venv/bin/activate # From the repository root
pip install -r requirements.txt
python plots.py          # data figures -> outputs/1-4_*.png
python run.py            # all methods + analysis (~10 min on CPU, faster on Apple GPU)
python -m pytest -q      # unit tests: physics, labels, peak finding, matching, AP
```
Change any value in `config.py` and re-run.

**Reproducibility:** every dataset uses its own seed (`config.seeds`); the GBM and PyTorch are seeded. CPU runs are bit-identical; Apple-GPU (MPS) runs differ slightly because those kernels are not deterministic.

## Files
| File | Contents |
|---|---|
| `config.py` | All knobs, plus the `HARD` stress config |
| `generate.py` | Dipole physics, survey simulation, labels |
| `plots.py` | Data gallery, signal decomposition, physics check, amplitude overlap |
| `evaluate.py` | Peak finding, matching, AP, recall @ FA budget |
| `baselines.py` | Rung 1 (amplitude, matched filter) and Rung 2 (features + GBM) |
| `cnn.py` | Rung 3 (U-Net, focal loss, training) |
| `analysis.py` | Recall vs depth, one-knob sweep, failure gallery |
| `run.py` | One command for everything |
| `tests/test_core.py` | Unit tests |