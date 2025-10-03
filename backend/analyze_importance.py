# analyze_importance.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

def choose_model_path(save_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    metrics_path = save_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            best_name, best_score = None, -1
            for name in ["xgboost", "random_forest", "logreg"]:
                m = metrics.get(name)
                if isinstance(m, dict):
                    score = m.get("roc_auc")
                    if score is None:
                        score = m.get("acc", -1)
                    if score is not None and score > best_score:
                        best_score = score
                        best_name = name
            if best_name:
                fname = f"model_{best_name}.joblib"
                cand = save_dir / fname
                if cand.exists():
                    return str(cand)
        except Exception:
            pass
    for fname in ["model_xgboost.joblib", "model_random_forest.joblib", "model_logreg.joblib"]:
        p = save_dir / fname
        if p.exists():
            return str(p)
    raise FileNotFoundError("No saved model found in models_out/. Train first or pass --model.")

def unwrap_estimator(est):
    try:
        from sklearn.calibration import CalibratedClassifierCV
        if isinstance(est, CalibratedClassifierCV):
            return est.calibrated_classifiers_[0].estimator
    except Exception:
        pass
    return est

def get_feature_names_from_preprocessor(pre):
    try:
        return np.array(pre.get_feature_names_out())
    except Exception:
        pass

    names = []
    try:
        for name, trans, cols in pre.transformers_:
            if name == "remainder":
                continue
          
            if hasattr(trans, "named_steps"):
              
                ohe = None
                for key in ["onehot", "ohe"]:
                    if key in trans.named_steps:
                        ohe = trans.named_steps[key]
                        break
                if ohe is not None and hasattr(ohe, "get_feature_names_out"):
                    try:
                        ohe_names = ohe.get_feature_names_out(cols)
                    except Exception:
                        ohe_names = [f"{c}" for c in cols]
                    names.extend([f"{name}__{n}" for n in ohe_names])
                else:
                    names.extend([f"{name}__{c}" for c in cols])
            else:
                names.extend([f"{name}__{c}" for c in cols])
        return np.array(names)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Export feature importances from the saved pipeline.")
    ap.add_argument("--model", default=None, help="Path to a saved model .joblib. If omitted, auto-pick from models_out/.")
    ap.add_argument("--save_dir", default="models_out", help="Directory where models and outputs live.")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = choose_model_path(save_dir, args.model)
    pipe = joblib.load(model_path)

    try:
        pre = pipe.named_steps["prep"]
        clf = pipe.named_steps["clf"]
    except Exception:
        raise RuntimeError("Loaded object is not the expected Pipeline with 'prep' and 'clf' steps.")

    inner = unwrap_estimator(clf)

    feature_names = get_feature_names_from_preprocessor(pre)
    if feature_names is None:
        n_features = None
        if hasattr(inner, "n_features_in_"):
            n_features = int(inner.n_features_in_)
        elif hasattr(inner, "coef_"):
            n_features = int(np.array(inner.coef_).shape[-1])
        elif hasattr(inner, "feature_importances_"):
            n_features = int(len(inner.feature_importances_))
        if n_features is None:
            raise RuntimeError("Could not infer feature names or count.")
        feature_names = np.array([f"feature_{i}" for i in range(n_features)])

    importances = None
    model_kind = type(inner).__name__

    if hasattr(inner, "feature_importances_"):
        importances = np.array(inner.feature_importances_, dtype=float)
    
    elif hasattr(inner, "coef_"):
        coefs = np.asarray(inner.coef_, dtype=float)
        coefs = np.squeeze(coefs)  
        if coefs.ndim > 1:
           
            coefs = np.linalg.norm(coefs, axis=0)
        importances = np.abs(coefs)

    if importances is None:
        raise RuntimeError(f"Estimator '{model_kind}' does not expose feature importances or coefficients.")

    
    if len(importances) != len(feature_names):
        m = min(len(importances), len(feature_names))
        importances = importances[:m]
        feature_names = feature_names[:m]

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "model": model_kind,
        "model_path": model_path,
    }).sort_values("importance", ascending=False)

    out_path = save_dir / "feature_importance.csv"
    df.to_csv(out_path, index=False)

    print(f"Loaded model: {model_path}")
    print(f"Estimator: {model_kind}")
    print("Top 15 features:")
    print(df.head(15).to_string(index=False))
    print(f"\nWrote {out_path.resolve()}")

if __name__ == "__main__":
    main()
