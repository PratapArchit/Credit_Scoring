import warnings
warnings.filterwarnings("ignore")

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from credit_score_pipeline import engineer_features, load_model

def choose_model_path(save_dir: str) -> str:
    save_dir = Path(save_dir)
    metrics_path = save_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            best_name, best_score = None, -1
            for name in ["logreg", "random_forest", "xgboost"]:
                m = metrics.get(name)
                if isinstance(m, dict):
                    score = m.get("roc_auc") if m.get("roc_auc") is not None else m.get("acc", -1)
                    if score is not None and score > best_score:
                        best_score = score
                        best_name = name
            if best_name:
                cand = save_dir / f"model_{'xgboost' if best_name=='xgboost' else best_name}.joblib"
                if cand.exists():
                    return str(cand)
        except Exception:
            pass
    
    for fn in ["model_logreg.joblib", "model_random_forest.joblib", "model_xgboost.joblib"]:
        p = save_dir / fn
        if p.exists():
            return str(p)
    raise FileNotFoundError("No saved model found in save_dir. Train first.")

def _get_pipeline_parts(pipe):
    """Return (preprocessor, estimator) from a sklearn Pipeline"""
    try:
        pre = pipe.named_steps.get("prep", None)
        clf = pipe.named_steps.get("clf", None)
        return pre, clf
    except Exception:
        return None, pipe  

def _pretty_names(pre, raw_feature_names=None):
    """Get readable feature names from ColumnTransformer if available."""
    try:
        names = pre.get_feature_names_out()
        return [n.replace("num__", "").replace("bin__", "").replace("cat__", "") for n in names]
    except Exception:
        return list(raw_feature_names) if raw_feature_names is not None else None

def main():
    ap = argparse.ArgumentParser(description="Global explainability using SHAP (with graceful fallback).")
    ap.add_argument("--csv", default="data.csv", help="Input CSV used to build the background matrix.")
    ap.add_argument("--model", default=None, help="Path to saved model .joblib. If omitted, auto-pick from save_dir.")
    ap.add_argument("--save_dir", default="models_out", help="Where models/metrics live; outputs are written here.")
    ap.add_argument("--max_rows", type=int, default=2000, help="Sample up to N rows for speed.")
    args = ap.parse_args()

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model or choose_model_path(args.save_dir)
    pipe = load_model(model_path)
    pre, clf = _get_pipeline_parts(pipe)

    raw = pd.read_csv(args.csv)
    if args.max_rows and len(raw) > args.max_rows:
        raw = raw.sample(args.max_rows, random_state=42).reset_index(drop=True)
    X = engineer_features(raw)

    if pre is None or clf is None:
        Xp = X.values
        feature_names = list(X.columns)
        est = pipe
    else:
        Xp = pre.transform(X)
        feature_names = _pretty_names(pre)
        est = clf

    try:
        import scipy.sparse as sp
        is_sparse = sp.issparse(Xp)
    except Exception:
        is_sparse = False
    Xp_dense = Xp.toarray() if is_sparse else Xp

    if HAS_SHAP:
        try:
            est_name = est.__class__.__name__.lower()
            if ("forest" in est_name) or ("xgb" in est_name) or ("gradientboosting" in est_name):
                explainer = shap.TreeExplainer(est)
            elif "logisticregression" in est_name:
                explainer = shap.LinearExplainer(est, Xp_dense)
            else:
                explainer = shap.Explainer(est, Xp_dense)

            shap_values = explainer(Xp_dense)
            sv = getattr(shap_values, "values", shap_values)
            if sv.ndim == 3 and sv.shape[2] >= 2:
                sv = sv[:, :, 1]

            mag = np.mean(np.abs(sv), axis=0)
            imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mag}).sort_values(
                "mean_abs_shap", ascending=False
            )
            imp.to_csv(out_dir / "shap_importance.csv", index=False)

            import matplotlib.pyplot as plt
            plt.figure()
            shap.summary_plot(sv, features=Xp_dense, feature_names=feature_names, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(out_dir / "shap_beeswarm_named.png", dpi=200)
            plt.close()

            try:
                plt.figure()
                shap.summary_plot(sv, features=Xp_dense, feature_names=feature_names, show=False, plot_type="bar", max_display=20)
                plt.tight_layout()
                plt.savefig(out_dir / "shap_bar_top20.png", dpi=200)
                plt.close()
            except Exception:
                pass

            print(f"Saved: {out_dir/'shap_importance.csv'}, {out_dir/'shap_beeswarm_named.png'}")
            return
        except Exception as e:
            print(f"[WARN] SHAP import succeeded but plotting/computation failed: {e}. Falling back to model-native importances.")
    importances = None
    label = None
    try:
        if hasattr(est, "feature_importances_"):
            importances = np.asarray(est.feature_importances_, dtype=float)
            label = "tree_feature_importance"
        elif hasattr(est, "coef_"):
            coef = est.coef_
            if coef is not None and coef.ndim == 2:
                importances = np.abs(coef[0]).astype(float)
                label = "abs_logreg_coeff"
    except Exception:
        importances = None

    if importances is not None and feature_names is not None and len(importances) == len(feature_names):
        df = pd.DataFrame({"feature": feature_names, label: importances})
        if label == "tree_feature_importance":
            df = df.sort_values(label, ascending=False)
        else:
            df = df.sort_values(label, ascending=False)
        out_csv = out_dir / "feature_importance_fallback.csv"
        df.to_csv(out_csv, index=False)
        print(f"SHAP not installed. Wrote fallback importances to: {out_csv}")
    else:
        print("Could not compute fallback importances (unknown model type). Install SHAP for full explanations.")

if __name__ == "__main__":
    main()
