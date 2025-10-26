import os, json, joblib
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from credit_score_pipeline import (
    train_all_models, load_model, predict_scores,
    engineer_features, weak_label_good_payer
)
DATA_PATH = "data.csv"
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded dataset: {df.shape}")
print("🚀 Training models using built-in backend pipeline…")
best_model_path, results = train_all_models(DATA_PATH, target_column=None, save_dir="models_out")
print(f"🏆 Best model: {best_model_path}")
model = load_model(best_model_path)
pred_df = predict_scores(model, df)
os.makedirs("plots_out", exist_ok=True)
pred_df.to_csv("plots_out/predicted_scores.csv", index=False)
if "good_payer" in df.columns:
    actual = df["good_payer"].astype(int)
else:
    actual = weak_label_good_payer(engineer_features(df))
pred_prob = pred_df["prob_good"]
r2  = r2_score(actual, pred_prob)
mae = mean_absolute_error(actual, pred_prob)
rmse = np.sqrt(mean_squared_error(actual, pred_prob))
corr, _ = pearsonr(actual, pred_prob)
with open("plots_out/regression_summary.txt","w") as f:
    f.write(f"R²={r2:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}\nCorr={corr:.4f}\n")
print(f"📈 R²={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  r={corr:.3f}")
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(actual, pred_prob, color="purple", alpha=0.6, edgecolors="k")
m,b=np.polyfit(actual,pred_prob,1)
plt.plot(actual,m*actual+b,color="red",lw=2,label="Regression Line")
plt.title("Regression Fit: Actual vs Predicted Probabilities")
plt.xlabel("Actual Good Payer"); plt.ylabel("Predicted Probability")
plt.legend(); plt.grid(True,ls="--",alpha=.5)

plt.subplot(1,2,2)
resid = pred_prob-actual
plt.scatter(actual,resid,color="teal",alpha=.5,edgecolors="k")
plt.axhline(0,color="red",ls="--")
plt.title("Residuals Plot")
plt.xlabel("Actual Good Payer"); plt.ylabel("Residual (Pred-Act)")
plt.grid(True,ls="--",alpha=.5)
plt.tight_layout()
plt.savefig("plots_out/regression_analysis.png",dpi=300,bbox_inches="tight")
plt.close()
print("\n📉 Calculating error-rates…")
actual_safe = actual.replace(0,np.nan)
percent_err = ((pred_prob-actual_safe).abs()/actual_safe)*100
percent_err = percent_err.fillna(0)
avg_pe = percent_err.mean()
rmse_rate = np.sqrt(np.mean(((pred_prob-actual_safe)/actual_safe)**2))*100
print(f"📊 Avg % Error Rate = {avg_pe:.2f}%")
print(f"📊 RMS Error Rate    = {rmse_rate:.2f}%")

with open("plots_out/error_rate_summary.txt","w") as f:
    f.write(f"Average Percentage Error Rate: {avg_pe:.2f}%\n")
    f.write(f"Average Root Mean Square Error Rate: {rmse_rate:.2f}%\n")
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(percent_err,bins=30,color="royalblue",alpha=.7,edgecolor="black")
plt.axvline(avg_pe,color="red",ls="--",lw=2,label=f"Mean = {avg_pe:.2f}%")
plt.title("Average Percentage Error Distribution")
plt.xlabel("Percentage Error (%)"); plt.ylabel("Frequency")
plt.legend(); plt.grid(True,ls="--",alpha=.5)

plt.subplot(1,2,2)
plt.bar(["RMSE Rate"],[rmse_rate],color="purple",alpha=.7)
plt.text(0,rmse_rate+0.5,f"{rmse_rate:.2f}%",ha="center",fontsize=10,fontweight="bold")
plt.title("Root Mean Square Error Rate")
plt.ylabel("Error Rate (%)"); plt.grid(axis="y",ls="--",alpha=.5)
plt.tight_layout()
plt.savefig("plots_out/error_rate_analysis.png",dpi=300,bbox_inches="tight")
plt.close()
print("✅ Error-rate analysis complete.  Outputs in plots_out/")
