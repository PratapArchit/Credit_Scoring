import re
import math
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

HAS_XGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False


YES_WORDS = {"yes", "y", "true", "1"}
NO_WORDS  = {"no", "n", "false", "0", ""}

def yesno(x: str) -> int:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0
    s = str(x).strip().lower()
    return 1 if s in YES_WORDS or "yes" in s else 0

def to_number_or_nan(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    s = re.sub(r"[^\d.]+", "", str(x))
    try:
        return float(s) if s != "" else np.nan
    except Exception:
        return np.nan

def months_value(x):
    v = to_number_or_nan(x)
    if pd.isna(v): return np.nan
    return float(v)

def ratio(numer, denom):
    if pd.isna(numer) or pd.isna(denom) or denom == 0: return np.nan
    return float(numer) / float(denom)

def clip01(x):
    if pd.isna(x): return np.nan
    return max(0.0, min(1.0, float(x)))

def clipmax(x, m=3.0):
    if pd.isna(x): return np.nan
    try:
        return min(float(x), float(m))
    except Exception:
        return x

def map_ontime(x):
    if pd.isna(x): return 1
    s = str(x).strip().lower()
    if "always" in s:    return 2
    if "sometimes" in s: return 1
    if "never" in s:     return 0
    return 1

def city_tier(city):
    tier1 = {"mumbai","delhi","new delhi","bangalore","bengaluru"}
    tier2 = {"ahmedabad","pune","hyderabad","chennai","kolkata"}
    if pd.isna(city): return "tier3"
    c = str(city).strip().lower()
    if c in tier1: return "tier1"
    if c in tier2: return "tier2"
    return "tier3"

def phone_price_tier(x):
    if pd.isna(x): return "Mid-Range"
    s = str(x).strip().lower()
    if "budget" in s: return "Budget"
    if "flagship" in s: return "Flagship"
    return "Mid-Range"


def engineer_features(df: pd.DataFrame):
    COL = {
        "age": "Age",
        "city": "City",
        "is_student": "Are you currently a student ?",
        "education": "What is your highest/current education level ?",
        "is_employed": "Are you currently employed ?",
        "income": "If Yes, what is your monthly income ?",
        "upi_use": "Do you use UPI ?",
        "upi_txn_week": "If yes, how many UPI transactions do you make in a week?",
        "monthly_spend": "What is your average monthly spending ?",
        "pay_rent": "Do you pay rent ?",
        "rent_amt": "If yes, how much you pay monthly ?",
        "rent_mode": "How do you pay your rent ?",
        "bills_ontime": "Do you pay bills (electricity, mobile, internet) on time?",
        "recharge_spend": "How much do you spend on mobile recharges monthly?",
        "subs_count": "How many subscriptions do you pay for?",
        "subs_list": "List your paid subscriptions",
        "subs_late": "How often are you unable to pay subscriptions on time?",
        "phone_type": "What phone type do you use?",
        "phone_model": "Phone brand and model ?",
        "phone_hours": "How many hours/day do you use your phone?",
        "do_save": "Do you actively save a portion of your income?",
        "save_amt": "If yes, how much do you save per month",
        "save_where": "If you save money, where do you keep your savings ?",
        "use_saving_apps": "Do you use savings tracking applications ?",
        "fin_apps": "Which financial apps do you use regularly?",
        "has_emi": "Do you have any active EMIs or loans (student loan, mobile EMI, etc.)?",
        "emi_total": "If yes, total EMI amount paid per month?",
        "used_bnpl": "Have you ever used Buy Now Pay Later (BNPL) services (ZestMoney, LazyPay, Simpl)?",
        "has_goal": "Do you have a financial goal you're saving for?",
        "goal_name": "If yes, what is the goal?",
        "goal_months": "How soon do you aim to achieve this goal (months)?",
        "emergency_fund": "Do you have an emergency fund that can cover 3+ months of expenses?",
        "emergency_how": "If faced with an emergency, how would you manage funds?",
        "crypto": "Do you invest in crypto ?",
        "invest_which": "Do you invest in any of the following ?",
        "charity": "Do you participate in charity donations or volunteering?",
        "address_months": "How long have you been living at your current address (months) ?",
        "job_months": "How long have you been at your current job/college (months)?",
    }

    X = pd.DataFrame(index=df.index)

    X["age"] = pd.to_numeric(df[COL["age"]], errors="coerce")
    X["income"] = df[COL["income"]].apply(to_number_or_nan)
    X["monthly_spend"] = df[COL["monthly_spend"]].apply(to_number_or_nan)
    X["rent_amt"] = df[COL["rent_amt"]].apply(to_number_or_nan)
    X["upi_txn_week"] = pd.to_numeric(df[COL["upi_txn_week"]], errors="coerce")
    X["recharge_spend"] = df[COL["recharge_spend"]].apply(to_number_or_nan)
    X["subs_count"] = pd.to_numeric(df[COL["subs_count"]], errors="coerce")
    X["phone_hours"] = pd.to_numeric(df[COL["phone_hours"]], errors="coerce")
    X["save_amt"] = df[COL["save_amt"]].apply(to_number_or_nan)
    X["emi_total"] = df[COL["emi_total"]].apply(to_number_or_nan)
    X["goal_months"] = pd.to_numeric(df[COL["goal_months"]], errors="coerce")
    X["address_months"] = df[COL["address_months"]].apply(months_value)
    X["job_months"] = df[COL["job_months"]].apply(months_value)

    X["spend_to_income"] = X.apply(lambda r: clipmax(ratio(r["monthly_spend"], r["income"]), 3.0), axis=1)
    X["rent_to_income"]  = X.apply(lambda r: clipmax(ratio(r["rent_amt"], r["income"]), 3.0), axis=1)
    X["emi_to_income"]   = X.apply(lambda r: clipmax(ratio(r["emi_total"], r["income"]), 3.0), axis=1)
    X["save_rate"]       = X.apply(lambda r: clip01(ratio(r["save_amt"], r["income"])), axis=1)
    
    X["obligation_ratio"] = X.apply(lambda r: clipmax(ratio(r.get("rent_amt", np.nan) + r.get("emi_total", np.nan), r["income"]), 3.0), axis=1)

    X["is_student"] = df[COL["is_student"]].apply(yesno)
    X["is_employed"] = df[COL["is_employed"]].apply(yesno)
    X["upi_use"] = df[COL["upi_use"]].apply(yesno)
    X["pay_rent"] = df[COL["pay_rent"]].apply(yesno)
    X["do_save"] = df[COL["do_save"]].apply(yesno)
    X["use_saving_apps"] = df[COL["use_saving_apps"]].apply(yesno)
    X["has_emi"] = df[COL["has_emi"]].apply(yesno)
    X["used_bnpl"] = df[COL["used_bnpl"]].apply(yesno)
    X["has_goal"] = df[COL["has_goal"]].apply(yesno)
    X["emergency_fund"] = df[COL["emergency_fund"]].apply(yesno)
    X["crypto"] = df[COL["crypto"]].apply(yesno)
    X["charity"] = df[COL["charity"]].apply(yesno)

    
    X["city_tier"] = df[COL["city"]].apply(city_tier)
    X["education"] = df[COL["education"]].fillna("Unknown")
    X["rent_mode"] = df[COL["rent_mode"]].fillna("None")
    X["phone_type"] = df[COL["phone_type"]].apply(phone_price_tier)
    X["bills_ontime"] = df[COL["bills_ontime"]].apply(map_ontime)  
    X["subs_late"] = df[COL["subs_late"]].apply(map_ontime)        

    fin_apps = df[COL["fin_apps"]].astype(str).str.lower()
    X["uses_jar"] = fin_apps.apply(lambda s: 1 if "jar" in s else 0)
    X["uses_gpay"] = fin_apps.apply(lambda s: 1 if "gpay" in s else 0)
    X["uses_paytm"] = fin_apps.apply(lambda s: 1 if "paytm" in s else 0)
    X["uses_fi"] = fin_apps.apply(lambda s: 1 if re.search(r"\bfi\b", s) else 0)
    X["uses_phonespe"] = fin_apps.apply(lambda s: 1 if "phonepe" in s else 0)  
    X["uses_groww"] = fin_apps.apply(lambda s: 1 if "groww" in s else 0)
    X["uses_coinswitch"] = fin_apps.apply(lambda s: 1 if "coinswitch" in s else 0)

    invest_txt = df[COL["invest_which"]].astype(str).str.lower()
    X["invest_stocks"] = invest_txt.apply(lambda s: 1 if "stock" in s else 0)
    X["invest_mf"] = invest_txt.apply(lambda s: 1 if "mutual fund" in s else 0)
    X["invest_fd"] = invest_txt.apply(lambda s: 1 if "fixed deposit" in s or "fixed deposits" in s else 0)
    X["invest_savings"] = invest_txt.apply(lambda s: 1 if "savings" in s else 0)

    for c in ["upi_txn_week","subs_count","phone_hours","recharge_spend"]:
        X[c] = X[c].fillna(0)

    return X

def compute_risk_points(X: pd.DataFrame) -> pd.Series:
    pts = pd.Series(0.0, index=X.index)

    pts += np.where(X["spend_to_income"] > 1.0, 4,
            np.where(X["spend_to_income"] > 0.7, 2, 0))

    pts += np.where(X["emi_to_income"] > 0.5, 3,
            np.where(X["emi_to_income"] > 0.3, 2,
            np.where(X["emi_to_income"] > 0.1, 1, 0)))

    pts += np.where(X["rent_to_income"] > 0.4, 2,
            np.where(X["rent_to_income"] > 0.25, 1, 0))

    pts += np.where(X["used_bnpl"]==1, 3, 0)

    pts += np.where(X["bills_ontime"]==0, 3,
            np.where(X["bills_ontime"]==1, 1, 0))

    pts += np.where(X["subs_late"]==2, 2,
            np.where(X["subs_late"]==1, 1, 0))

    pts += np.where((X["subs_count"] >= 3) & (X["income"] < 20000), 1, 0)

    pts += np.where(X["upi_txn_week"] >= 20, 1, 0)

    pts += np.where(X["save_rate"] < 0.05, 2,
            np.where(X["save_rate"] < 0.10, 1, 0))

    pts += np.where(X["emergency_fund"]==0, 2, 0)

    pts += np.where(X["crypto"]==1, 1, 0)

    pts += np.where(X["job_months"].fillna(0) < 12, 1, 0)
    pts += np.where(X["address_months"].fillna(0) < 6, 1, 0)
    pts += np.where((X["phone_hours"] >= 8), 1, 0)

    return pts

def weak_label_good_payer(X: pd.DataFrame) -> pd.Series:
    pts = compute_risk_points(X)
    return (pts <= 3).astype(int)

def prob_to_cibil(prob_good: np.ndarray) -> np.ndarray:
    return 300.0 + prob_good * 600.0


def make_bands(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        return "Unknown"
    if s >= 780: return "A (Prime)"
    if s >= 720: return "B (Good)"
    if s >= 660: return "C (Fair)"
    if s >= 600: return "D (Subprime)"
    return "E (High Risk)"

def attach_reasons(row: pd.Series, top_k: int = 3) -> str:
    reasons = []

    try:
        if row.get("emi_to_income", 0) >= 0.4: reasons.append("High EMI-to-income")
        if row.get("spend_to_income", 0) >= 0.7: reasons.append("High spend-to-income")
        if row.get("bills_ontime", 2) <= 1: reasons.append("Irregular bill payments")
        if row.get("subs_late", 2) <= 1: reasons.append("Subscription payments irregular")
        if row.get("emergency_fund", 1) == 0: reasons.append("No emergency fund")
        if row.get("do_save", 1) == 0: reasons.append("No regular savings")
        if row.get("used_bnpl", 0) == 1: reasons.append("BNPL usage indicates higher risk")
        if row.get("save_rate", 1) <= 0.05: reasons.append("Very low savings rate")
        if row.get("obligation_ratio", np.nan) >= 0.6: reasons.append("High fixed obligations")
    except Exception:
        pass
   
    pos = []
    try:
        if row.get("bills_ontime", 0) == 2: pos.append("Always pays bills on time")
        if row.get("subs_late", 0) == 2: pos.append("Subscriptions paid on time")
        if row.get("save_rate", 0) >= 0.2: pos.append("Healthy savings rate")
        if row.get("do_save", 0) == 1: pos.append("Regular saver")
        if row.get("emergency_fund", 0) == 1: pos.append("Has emergency fund")
        if row.get("emi_to_income", 1) <= 0.2: pos.append("Low EMI burden")
        if row.get("spend_to_income", 1) <= 0.5: pos.append("Moderate spending vs income")
    except Exception:
        pass
    
    final = reasons[:2] + pos[:1]
    return "; ".join(final) if final else "No strong risk signals detected"


def train_all_models(csv_path: str, target_column: str = None, save_dir: str = "models_out"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(csv_path)
    X_all = engineer_features(raw)

    if target_column and target_column in raw.columns:
        y_all = raw[target_column].astype(int).clip(0,1)
        label_source = "supervised target"
    else:
        y_all = weak_label_good_payer(X_all)
        label_source = "weak label (heuristic)"

    numeric_cols = [
        "age","income","monthly_spend","rent_amt","upi_txn_week","recharge_spend",
        "subs_count","phone_hours","save_amt","emi_total","goal_months",
        "address_months","job_months","spend_to_income","rent_to_income",
        "emi_to_income","save_rate","obligation_ratio","bills_ontime","subs_late"
    ]
    binary_cols = [
        "is_student","is_employed","upi_use","pay_rent","do_save","use_saving_apps",
        "has_emi","used_bnpl","has_goal","emergency_fund","crypto","charity",
        "uses_jar","uses_gpay","uses_paytm","uses_fi","uses_phonespe","uses_groww","uses_coinswitch",
        "invest_stocks","invest_mf","invest_fd","invest_savings"
    ]
    cat_cols = ["city_tier","education","rent_mode","phone_type"]

 
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("bin", binary_transformer, binary_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_all[numeric_cols + binary_cols + cat_cols],
        y_all,
        test_size=0.35,
        random_state=42,
        stratify=y_all if y_all.nunique() == 2 else None
    )

    results = {}

    lr = Pipeline(steps=[
        ("prep", pre),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:,1]
    lr_pred  = (lr_probs >= 0.5).astype(int)
    results["logreg"] = {
        "acc": accuracy_score(y_test, lr_pred),
        "roc_auc": roc_auc_score(y_test, lr_probs) if y_test.nunique()==2 else None,
        "f1": f1_score(y_test, lr_pred) if y_test.nunique()==2 else None,
        "cm": confusion_matrix(y_test, lr_pred).tolist()
    }
    joblib.dump(lr, save_path / "model_logreg.joblib")

    rf = Pipeline(steps=[
        ("prep", pre),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:,1]
    rf_pred  = (rf_probs >= 0.5).astype(int)
    results["random_forest"] = {
        "acc": accuracy_score(y_test, rf_pred),
        "roc_auc": roc_auc_score(y_test, rf_probs) if y_test.nunique()==2 else None,
        "f1": f1_score(y_test, rf_pred) if y_test.nunique()==2 else None,
        "cm": confusion_matrix(y_test, rf_pred).tolist()
    }
    joblib.dump(rf, save_path / "model_random_forest.joblib")

    if HAS_XGB:
        xgb = Pipeline(steps=[
            ("prep", pre),
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                objective="binary:logistic",
                random_state=42,
                eval_metric="logloss",
                tree_method="hist"
            ))
        ])
        xgb.fit(X_train, y_train)
        xgb_probs = xgb.predict_proba(X_test)[:,1]
        xgb_pred  = (xgb_probs >= 0.5).astype(int)
        results["xgboost"] = {
            "acc": accuracy_score(y_test, xgb_pred),
            "roc_auc": roc_auc_score(y_test, xgb_probs) if y_test.nunique()==2 else None,
            "f1": f1_score(y_test, xgb_pred) if y_test.nunique()==2 else None,
            "cm": confusion_matrix(y_test, xgb_pred).tolist()
        }
        joblib.dump(xgb, save_path / "model_xgboost.joblib")
    else:
        results["xgboost"] = "xgboost not installed"

    print("\n=== Label source:", label_source, "===")
    for k, v in results.items():
        print(f"\n{k} -> {v}")

    with open(save_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    best_model = "model_random_forest.joblib"
    try:
        scores = []
        for name in ["logreg","random_forest","xgboost"]:
            if isinstance(results.get(name), dict):
                s = results[name]["roc_auc"] if results[name]["roc_auc"] is not None else results[name]["acc"]
                scores.append((name, s))
        scores.sort(key=lambda t: t[1], reverse=True)
        if scores:
            best = scores[0][0]
            best_model = f"model_{'xgboost' if best=='xgboost' else best}.joblib"
    except Exception:
        pass

    print(f"\nSaved models to: {save_path.resolve()}")
    print(f"Best model (by AUC/Acc): {best_model}")
    return str(save_path / best_model), results



def load_model(model_path: str):
    return joblib.load(model_path)

def predict_scores(model, df_raw: pd.DataFrame):
    X = engineer_features(df_raw)
    proba_good = model.predict_proba(X)[:,1]
    score = prob_to_cibil(proba_good)
    out = df_raw.copy()
    out["prob_good"] = proba_good
    out["cibil_like_score"] = score.round(0).clip(300, 900)
    out["score_band"] = out["cibil_like_score"].apply(make_bands)

    try:
        out = out.reset_index(drop=True)
        out["top_reasons"] = X.apply(attach_reasons, axis=1)
    except Exception:
        out["top_reasons"] = ""
    return out

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train credit scoring models and output CIBIL-like scores.")
    parser.add_argument("--csv", required=True, help="Path to your CSV (with the exact columns).")
    parser.add_argument("--target", default=None, help="Optional supervised binary target column (1=good,0=bad).")
    parser.add_argument("--save_dir", default="models_out", help="Where to save models/metrics.")
    parser.add_argument("--predict_only", default=None, help="If provided, skip training and predict with the given model path.")
    args = parser.parse_args()

    if args.predict_only:
        mdl = load_model(args.predict_only)
        raw_df = pd.read_csv(args.csv)
        scored = predict_scores(mdl, raw_df)
        scored.to_csv("scored_output.csv", index=False)
        print("Wrote scored_output.csv")
    else:
        best_model_path, _ = train_all_models(args.csv, target_column=args.target, save_dir=args.save_dir)
        mdl = load_model(best_model_path)
        raw_df = pd.read_csv(args.csv)
        scored = predict_scores(mdl, raw_df)
        out_path = Path(args.save_dir) / "scored_output.csv"
        scored.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

