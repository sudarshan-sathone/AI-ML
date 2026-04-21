import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    classification_report,
)

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "D:\\AI-ML\\Programs\\Day_13\\dataset.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "D:\\AI-ML\\Programs\\Day_13\\outputs\\results.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Dataset ──────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

FEATURE_COLS = [
    "skill_diff",
    "common_tech",
    "availability_match",
    "communication_score",
    "experience_gap",
]
TARGET_COL = "compatibility"

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

print("Class distribution:\n", df[TARGET_COL].value_counts(), "\n")

# ── Train/Test Split ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── Polynomial Features (interaction terms only) ──────────────────
poly = PolynomialFeatures(
    degree=2,
    include_bias=False,
    interaction_only=True,
)
X_train_poly = poly.fit_transform(X_train)
X_test_poly  = poly.transform(X_test)

feature_names = poly.get_feature_names_out(FEATURE_COLS)

# ── Random Forest (optimized) ─────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train_poly, y_train)

# ── Predictions ───────────────────────────────────────────────────
y_pred = model.predict(X_test_poly)

# ── Metrics ───────────────────────────────────────────────────────
accuracy   = accuracy_score(y_test, y_pred)
recall     = recall_score(y_test, y_pred)
f1         = f1_score(y_test, y_pred)
conf_mat   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_mat.ravel()
cls_report = classification_report(
    y_test, y_pred,
    target_names=["Not Compatible (0)", "Compatible (1)"],
)

# ── Feature Importance ────────────────────────────────────────────
feat_df = (
    pd.DataFrame({
        "Feature":    feature_names,
        "Importance": model.feature_importances_,
    })
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

# ── Display Results ───────────────────────────────────────────────
sep = "=" * 60

print(sep)
print("EVALUATION METRICS")
print(sep)
print(f"  Accuracy : {accuracy * 100:.2f}%")
print(f"  Recall   : {recall * 100:.2f}%")
print(f"  F1 Score : {f1 * 100:.2f}%\n")

print(sep)
print("CONFUSION MATRIX")
print(sep)
print("  Rows = Actual | Columns = Predicted\n")
print(f"  {conf_mat}\n")
print(f"  TN: {tn}  FP: {fp}")
print(f"  FN: {fn}  TP: {tp}\n")

print(sep)
print("CLASSIFICATION REPORT")
print(sep)
print(cls_report)

print(sep)
print("TOP 10 FEATURE IMPORTANCE")
print(sep)
for _, row in feat_df.head(10).iterrows():
    bar = "X" * int(row["Importance"] * 40)
    print(f"  {row['Feature']:<45} {row['Importance']:.4f}  {bar}")

# ── Save Results ──────────────────────────────────────────────────
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("PROJECT TEAMMATE COMPATIBILITY PREDICTION - RESULTS\n")
    f.write(sep + "\n\n")

    f.write("EVALUATION METRICS\n")
    f.write(sep + "\n")
    f.write(f"  Accuracy : {accuracy * 100:.2f}%\n")
    f.write(f"  Recall   : {recall * 100:.2f}%\n")
    f.write(f"  F1 Score : {f1 * 100:.2f}%\n\n")

    f.write("CONFUSION MATRIX\n")
    f.write(sep + "\n")
    f.write(f"  {conf_mat}\n\n")
    f.write(f"  TN: {tn}  FP: {fp}\n")
    f.write(f"  FN: {fn}  TP: {tp}\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write(sep + "\n")
    f.write(cls_report + "\n")

    f.write("TOP 10 FEATURE IMPORTANCE\n")
    f.write(sep + "\n")
    for _, row in feat_df.head(10).iterrows():
        bar = "X" * int(row["Importance"] * 40)
        f.write(f"  {row['Feature']:<45} {row['Importance']:.4f}  {bar}\n")

    f.write("\nFULL FEATURE IMPORTANCE TABLE\n")
    f.write(sep + "\n")
    f.write(feat_df.to_string(index=False))
    f.write("\n")

print(f"\nResults saved to: {RESULTS_PATH}")
