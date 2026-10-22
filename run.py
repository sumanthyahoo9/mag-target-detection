"""
One command to run the entire pipeline:
generate data -> 
run each method -> 
evaluate on test + HARD
analysis.
Run the following command to retrain the U-Net on HARD data (Physics ceiling vs Distribution shift)
`python run.py --train-hard`
"""
import sys

from config import HARD, Config
from analysis import failure_gallery, knob_sweep, recall_by_depth
from cnn import predict, train_cnn
from evaluate import evaluate
from generate import make_dataset
from baselines import (amplitude_scores, blob_scores, fit_blob_classifier,
                       matched_filter_scores, scale_to_unit)


def report(method_name: str, results: dict) -> None:
    """
    Report scores
    """
    ap = f"AP {results['ap']:.3f}"
    budget = f"recall @ <=0.5 FA/patch {results['recall_at_fa']:.3f}"
    operating = f"thr 0.5: recall {results['recall_op']:.3f}, FA/patch {results['fa_per_patch_op']:.2f}"
    print(f"{method_name:<24} {ap} | {budget} | {operating}")


if __name__ == "__main__":
    cfg = Config()
    s_train, s_val, s_test, s_hard = cfg.seeds
    X_train, Y_train, T_train, _ = make_dataset(cfg, cfg.n_train, s_train)
    X_val, _, T_val, _ = make_dataset(cfg, cfg.n_val, s_val)
    X_test, _, T_test, B_test = make_dataset(cfg, cfg.n_test, s_test)
    X_hard, _, T_hard, _ = make_dataset(HARD, cfg.n_test, s_hard)
    splits = [("test", X_test, T_test), ("HARD", X_hard, T_hard)]

    # Rung 1: no learning. Calibration scale from validation only.
    for name, score_fn in [("Amplitude", amplitude_scores), ("Matched filter", matched_filter_scores)]:
        ref = score_fn(X_val, cfg)
        for split, X, T in splits:
            report(f"{name} [{split}]", evaluate(scale_to_unit(score_fn(X, cfg), ref), T, cfg))

    # Rung 2: blob features + gradient boosting (trained on the train split only).
    mf_ref = matched_filter_scores(X_val, cfg)
    clf = fit_blob_classifier(X_train, T_train, cfg, mf_ref)
    gbm_test = evaluate(blob_scores(clf, X_test, cfg, mf_ref), T_test, cfg)
    report("Blob GBM [test]", gbm_test)
    report("Blob GBM [HARD]", evaluate(blob_scores(clf, X_hard, cfg, mf_ref), T_hard, cfg))

    # Rung 3: small U-Net (model selection on validation AP only).
    model = train_cnn(X_train, Y_train, X_val, T_val, cfg)
    S_test = predict(model, X_test)
    unet_test = evaluate(S_test, T_test, cfg)
    report("U-Net [test]", unet_test)
    report("U-Net [HARD]", evaluate(predict(model, X_hard), T_hard, cfg))

    # Item 7: analysis.
    print("\nRecall @ thr 0.5 by burial depth  [0-1 m, 1-2 m, 2-3 m]")
    for name, res in [("Blob GBM", gbm_test), ("U-Net", unet_test)]:
        print(f"  {name:<10}", " ".join(f"{r:.2f}" 
                                        for r in recall_by_depth(res["hits_op"], T_test)))
    print(f"\nU-Net AP with ONE knob set to its HARD value (default test AP {unet_test['ap']:.3f}):")
    for knob, ap in knob_sweep(lambda X: predict(model, X), cfg, cfg.n_val, s_hard).items():
        print(f"  {knob:<18} {ap:.3f}")
    failure_gallery(X_test, S_test, T_test, B_test, cfg, "outputs/5_failures.png")

    if "--train-hard" in sys.argv:              
        # same model, trained and validated on HARD data
        Xh_tr, Yh_tr, _, _ = make_dataset(HARD, cfg.n_train, s_train)
        Xh_val, _, Th_val, _ = make_dataset(HARD, cfg.n_val, s_val)
        model_h = train_cnn(Xh_tr, Yh_tr, Xh_val, Th_val, HARD)
        report("U-Net trained-HARD [HARD]", evaluate(predict(model_h, X_hard), T_hard, HARD))