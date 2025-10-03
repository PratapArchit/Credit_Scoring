# tune_threshold.py
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    brier_score_loss,
    roc_auc_score,
)

from credit_score_pipeline import (
    engineer_features,
    weak_label_good_payer,
    load_model,
)

def choose_model_path(save_dir: str) -> str:
    """
    Pick the best model using models_out/metrics.json if present,
    else fall back to common model filenames.
    """
    save_dir = Path(save_dir)
    metrics_path = save_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            best_name, best_score = None, -1
            for name in ["logreg", "random_forest", "xgboost"]:
                m = metrics.get(name)
                if isinstance(m, dict):
                    score = m.get("roc_auc")
                    if score is None:
                        score = m.get("acc", -1)
                    if score is not None and score > best_score:
                        best_score = score
                        best_name = name
            if best_name:
                fname = f"model_{'xgboost' if best_name=='xgboost' else best_name}.joblib"
                candidate = save_dir / fname
                if candidate.exists():
                    return str(candidate)
        except Exception:
            pass

    for fname in [
        "model_logreg.joblib",
        "model_random_forest.joblib",
        "model_xgboost.joblib",
    ]:
        p = save_dir / fname
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "Could not find a saved model. Train first or pass --model explicitly."
    )


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute KS statistic between positives and negatives over score distribution."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    order = np.argsort(s)
    y_sorted = y[order]
    n_pos = max((y_sorted == 1).sum(), 1)
    n_neg = max((y_sorted == 0).sum(), 1)
    cdf_pos = np.cumsum(y_sorted == 1) / n_pos
    cdf_neg = np.cumsum(y_sorted == 0) / n_neg
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def expected_profit(tp: int, fp: int, gain_tp: float = 1.0, loss_fp: float = -3.0) -> float:
    """
    Simple profit proxy:
      +gain_tp for every approved true good (TP),
      +loss_fp for every approved true bad (FP).
    """
    return tp * gain_tp + fp * loss_fp

def main():
    ap = argparse.ArgumentParser(description="Sweep decision thresholds and pick business-optimal one.")
    ap.add_argument("--csv", default="data.csv", help="Input CSV used for tuning (same schema as training).")
    ap.add_argument("--model", default=None, help="Path to saved model .joblib. If omitted, auto-pick best from save_dir.")
    ap.add_argument("--save_dir", default="models_out", help="Where models/metrics.json live; also where output CSV will be saved.")
    ap.add_argument("--target", default=None, help="Optional true label column (1=good, 0=bad). If absent, uses weak_label_good_payer.")
    ap.add_argument("--low", type=float, default=0.05, help="Threshold sweep start (inclusive).")
    ap.add_argument("--high", type=float, default=0.95, help="Threshold sweep end (inclusive).")
    ap.add_argument("--step", type=float, default=0.01, help="Threshold step size.")
    ap.add_argument("--gain_tp", type=float, default=1.0, help="Profit for approving a true good (TP).")
    ap.add_argument("--loss_fp", type=float, default=-3.0, help="Loss for approving a true bad (FP).")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model or choose_model_path(save_dir)
    pipe = load_model(model_path)

    raw = pd.read_csv(args.csv)
    X = engineer_features(raw)  

    if args.target and args.target in raw.columns:
        y = raw[args.target].astype(int).clip(0, 1).values
        label_source = f"supervised target '{args.target}'"
    else:
        y = weak_label_good_payer(X).values
        label_source = "weak label (heuristic)"

    proba = pipe.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else np.nan
    brier = brier_score_loss(y, proba)
    ks = ks_statistic(y, proba)

    thresholds = np.round(np.arange(args.low, args.high + 1e-9, args.step), 4)
    rows = []
    for th in thresholds:
        yhat = (proba >= th).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, yhat, average="binary", zero_division=0
        )
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        approved = max(int((yhat == 1).sum()), 0)
        approval_rate = float(approved) / len(y)
        bad_rate_on_approved = (fp / approved) if approved > 0 else np.nan
        exp_profit = expected_profit(tp, fp, args.gain_tp, args.loss_fp)

        rows.append(
            dict(
                threshold=th,
                approval_rate=approval_rate,
                precision=prec,
                recall=rec,
                f1=f1,
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
                bad_rate_on_approved=bad_rate_on_approved,
                expected_profit=exp_profit,
                auc=auc,
                brier=brier,
                ks=ks,
                label_source=label_source,
                model_path=model_path,
            )
        )

    df = pd.DataFrame(rows)
    out_path = save_dir / "threshold_sweep.csv"
    df.sort_values("expected_profit", ascending=False).to_csv(out_path, index=False)

    top_profit = df.sort_values("expected_profit", ascending=False).head(5)
    top_f1 = df.sort_values("f1", ascending=False).head(5)

    print(f"\nModel: {model_path}")
    print(f"Labels: {label_source}")
    print(f"AUC={auc:.4f}  Brier={brier:.4f}  KS={ks:.4f}")
    print("\nTop by expected_profit:")
    print(top_profit[["threshold", "expected_profit", "approval_rate", "bad_rate_on_approved", "precision", "recall", "f1"]])
    print("\nTop by F1:")
    print(top_f1[["threshold", "f1", "approval_rate", "bad_rate_on_approved", "precision", "recall"]])
    print(f"\nWrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
