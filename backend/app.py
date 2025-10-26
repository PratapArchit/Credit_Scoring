import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.credit_score_pipeline import (
    engineer_features,
    load_model,
    prob_to_cibil,
    make_bands, 
    attach_reasons,
)

app = FastAPI(title="Gen-Z Credit Scoring API", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAVE_DIR = Path("models_out")

def choose_model_path(save_dir: Path) -> str:
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
                cand = save_dir / f"model_{best_name}.joblib"
                if cand.exists():
                    return str(cand)
        except Exception:
            pass
    for fname in ["model_xgboost.joblib", "model_random_forest.joblib", "model_logreg.joblib"]:
        p = save_dir / fname
        if p.exists():
            return str(p)
    raise FileNotFoundError("No saved model found in models_out/.")

MODEL_PATH = choose_model_path(SAVE_DIR)
PIPELINE = load_model(MODEL_PATH)

@app.get("/")
def root():
    return {"ok": True, "message": "Gen-Z Credit Scoring API. See /docs, /health, /score, /score_batch."}

@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}

@app.post("/score")
async def score(req: Request) -> Dict[str, Any]:
    """
    Accept a free-form JSON dict from the frontend.
    We avoid strict typing so empty strings in numeric fields don't 422.
    """
    try:
        row = await req.json()
        if not isinstance(row, dict):
            raise ValueError("Body must be a JSON object")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    df_raw = pd.DataFrame([row])

    X = engineer_features(df_raw)
    proba = PIPELINE.predict_proba(X)[:, 1]
    score_val = float(prob_to_cibil(proba)[0])
    band = make_bands(score_val)
    reasons = attach_reasons(X.iloc[0])

    shap_pos, shap_neg = "", ""
    try:
        from per_applicant_explanations import unwrap_calibrated, get_feature_names, top_shap_text
        import shap  # type: ignore
        clf = PIPELINE.named_steps["clf"]
        pre = PIPELINE.named_steps["prep"]
        inner = unwrap_calibrated(clf)
        Xmat = pre.transform(X)
        feat_names = get_feature_names(pre)
        if feat_names is None:
            feat_names = np.array([f"feature_{i}" for i in range(Xmat.shape[1])])
        if hasattr(inner, "feature_importances_"):
            explainer = shap.TreeExplainer(inner)
            exp = explainer(Xmat)
            sv = getattr(exp, "values", exp)
        elif hasattr(inner, "coef_"):
            explainer = shap.LinearExplainer(inner, Xmat)
            sv = explainer.shap_values(Xmat)
        else:
            background = Xmat if Xmat.shape[0] < 200 else Xmat[:200]
            explainer = shap.KernelExplainer(lambda d: inner.predict_proba(d)[:,1], background)
            sv = explainer.shap_values(Xmat, nsamples=100)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        pos_txt, neg_txt = top_shap_text(sv[0], feat_names, k=3)
        shap_pos, shap_neg = pos_txt, neg_txt
    except Exception:
        pass

    return {
        "prob_good": float(proba[0]),
        "cibil_like_score": round(max(300, min(900, score_val))),
        "score_band": band,
        "top_reasons": reasons,
        "shap_top_positive": shap_pos,
        "shap_top_negative": shap_neg,
    }

@app.post("/score_batch")
async def score_batch(req: Request) -> Dict[str, Any]:
    """
    Accepts: { "applicants": [ { ... }, { ... } ] }
    Each applicant is a free-form dict.
    """
    try:
        body = await req.json()
        if not isinstance(body, dict) or "applicants" not in body or not isinstance(body["applicants"], list):
            raise ValueError("Body must be an object with an 'applicants' list.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    rows: List[Dict[str, Any]] = body["applicants"]
    if not rows:
        return {"results": []}

    df_raw = pd.DataFrame(rows)
    X = engineer_features(df_raw)
    proba = PIPELINE.predict_proba(X)[:, 1]
    scores = prob_to_cibil(proba)
    bands = [make_bands(float(s)) for s in scores]
    reasons = [attach_reasons(X.iloc[i]) for i in range(len(X))]

    out = []
    for i in range(len(X)):
        out.append({
            "prob_good": float(proba[i]),
            "cibil_like_score": int(round(max(300, min(900, float(scores[i]))))),
            "score_band": bands[i],
            "top_reasons": reasons[i],
        })
    return {"results": out}
