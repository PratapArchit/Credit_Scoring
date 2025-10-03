import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from credit_score_pipeline import (
    engineer_features,
    prob_to_cibil,
    make_bands,
    attach_reasons,
)


def choose_model_path(save_dir: Path, explicit: str | None) -> str:
    """
    Pick the best model using models_out/metrics.json if present,
    else fall back to common model filenames.
    """
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

def unwrap_calibrated(est):
    try:
        from sklearn.calibration import CalibratedClassifierCV
        if isinstance(est, CalibratedClassifierCV):
            return est.calibrated_classifiers_[0].estimator
    except Exception:
        pass
    return est

def get_feature_names(pre):
    try:
        return pre.get_feature_names_out()
    except Exception:
        pass
    try:
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "named_steps"):
                for key in ["onehot", "ohe"]:
                    if key in trans.named_steps:
                        ohe = trans.named_steps[key]
                        try:
                            oh_names = ohe.get_feature_names_out(cols)
                        except Exception:
                            oh_names = cols
                        names.extend([f"{name}__{n}" for n in oh_names])
                        break
                else:
                    names.extend([f"{name}__{c}" for c in cols])
            else:
                names.extend([f"{name}__{c}" for c in cols])
        return np.array(names, dtype=object)
    except Exception:
        return None

def prettify_feature_name(raw: str) -> str:
    """
    Turn 'num__spend_to_income' or 'cat__education_Bachelor' into readable labels.
    """
    if raw is None:
        return ""
    s = str(raw)
    if "__" in s:
        s = s.split("__", 1)[1]
    s = s.replace("_", " ")
    s = s.replace(" = ", "=")
    return s.strip()

def top_shap_text(shap_values_row, feature_names, k=3):
    """
    Return two strings: top positive and top negative SHAP contributors.
    """
    vals = np.array(shap_values_row).astype(float)
    idx_pos = np.argsort(-vals)[:k]  
    idx_neg = np.argsort(vals)[:k]   
    pos = [prettify_feature_name(feature_names[i]) for i in idx_pos]
    neg = [prettify_feature_name(feature_names[i]) for i in idx_neg]
    return "; ".join(pos), "; ".join(neg)

def main():
    ap = argparse.ArgumentParser(description="Create per-applicant explanations CSV (bands, reasons, optional SHAP).")
    ap.add_argument("--csv", required=True, help="Input CSV with raw fields (same schema as training).")
    ap.add_argument("--model", default=None, help="Optional explicit model path (.joblib).")
    ap.add_argument("--save_dir", default="models_out", help="Directory where models_out lives and output will be written.")
    ap.add_argument("--k", type=int, default=3, help="Top-K SHAP drivers to include for positive/negative.")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = choose_model_path(save_dir, args.model)
    pipe = joblib.load(model_path)

    try:
        pre = pipe.named_steps["prep"]
        clf = pipe.named_steps["clf"]
    except Exception as e:
        raise RuntimeError("Loaded object is not a Pipeline with 'prep' and 'clf' steps.") from e

    raw = pd.read_csv(args.csv)
    X = engineer_features(raw)

    proba = pipe.predict_proba(X)[:, 1]
    score = prob_to_cibil(proba)

    out = raw.copy()
    out = out.reset_index(drop=True)
    out["prob_good"] = proba
    out["cibil_like_score"] = np.round(score).clip(300, 900).astype(int)
    out["score_band"] = out["cibil_like_score"].apply(make_bands)

    reasons = []
    for i in range(len(X)):
        try:
            reasons.append(attach_reasons(X.iloc[i]))
        except Exception:
            reasons.append("")
    out["top_reasons"] = reasons

    shap_pos, shap_neg = [""] * len(X), [""] * len(X)
    try:
        import shap  # type: ignore

        inner = unwrap_calibrated(clf)
        Xmat = pre.transform(X)
        feat_names = get_feature_names(pre)
        if feat_names is None:
            n = Xmat.shape[1]
            feat_names = np.array([f"feature_{i}" for i in range(n)])


        explainer = None
        use_kernel = False
        if hasattr(inner, "feature_importances_"):
            explainer = shap.TreeExplainer(inner)
        elif hasattr(inner, "coef_"):
           
            try:
                explainer = shap.LinearExplainer(inner, Xmat, feature_dependence="independent")
            except Exception:
                # Older/newer SHAP signatures; fallback to default
                explainer = shap.LinearExplainer(inner, Xmat)
        else:
            use_kernel = True

        if use_kernel:
            background = shap.sample(Xmat, 100) if Xmat.shape[0] > 100 else Xmat
            explainer = shap.KernelExplainer(lambda d: inner.predict_proba(d)[:, 1], background)
            sv = explainer.shap_values(Xmat, nsamples=100) 
        else:
            try:
                exp = explainer(Xmat)
                sv = exp.values
            except Exception:
                sv = explainer.shap_values(Xmat)

        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]

        for i in range(len(X)):
            pos_txt, neg_txt = top_shap_text(sv[i], feat_names, k=args.k)
            shap_pos[i] = pos_txt
            shap_neg[i] = neg_txt

        out["shap_top_positive"] = shap_pos
        out["shap_top_negative"] = shap_neg

    except Exception:
        out["shap_top_positive"] = ""
        out["shap_top_negative"] = ""

    
    out_path = save_dir / "per_applicant_explanations.csv"
    out.to_csv(out_path, index=False)

    print(f"Model: {model_path}")
    print(f"Wrote {out_path.resolve()}")
    print("Columns added: prob_good, cibil_like_score, score_band, top_reasons, shap_top_positive, shap_top_negative")

if __name__ == "__main__":
    main()
