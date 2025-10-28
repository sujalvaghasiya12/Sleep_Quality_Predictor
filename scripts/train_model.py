# scripts/train_model.py
"""
Train the sleep-quality model with improved defaults, save artifacts and explanations.

Produces:
- models/sleep_model.joblib
- results/metrics.txt
- results/roc_curve.png
- results/feature_importances.csv
- results/feature_importances.png
- optionally: results/shap_summary.png (if shap available)
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# ---------------------------
# Helper: OneHotEncoder compat
# ---------------------------
def get_onehot_encoder():
    """Return OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

# ---------------------------
# Paths & folders
# ---------------------------
Path('results').mkdir(parents=True, exist_ok=True)
Path('models').mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv('data/sleep_data.csv')

# Ensure target is explicit: 'good' -> 1, 'poor' -> 0
if df['sleep_quality'].dtype == object:
    df['sleep_quality'] = df['sleep_quality'].map({'good': 1, 'poor': 0})
df = df.dropna(subset=['sleep_quality'])
df['sleep_quality'] = df['sleep_quality'].astype(int)

# ---------------------------
# Features & target
# ---------------------------
features = [
    'hours_of_sleep','screen_time','caffeine_intake','steps_walked','stress_level',
    'exercise_minutes','alcohol_units','ambient_light','weekday','bedtime_hour'
]
X = df[features]
y = df['sleep_quality']

# ---------------------------
# Train / test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Preprocessing
# ---------------------------
numeric_features = [
    'hours_of_sleep','screen_time','caffeine_intake','steps_walked','stress_level',
    'exercise_minutes','alcohol_units','bedtime_hour'
]
cat_features = ['ambient_light','weekday']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', get_onehot_encoder())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', cat_transformer, cat_features)
], remainder='drop')

# ---------------------------
# Model pipeline (improved)
# ---------------------------
pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight='balanced',   # handle class imbalance
        max_depth=12,
        n_jobs=-1
    ))
])

# ---------------------------
# Train
# ---------------------------
pipeline.fit(X_train, y_train)
print("Model trained.")

# ---------------------------
# Evaluate
# ---------------------------
y_pred = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

if probs is not None:
    auc = roc_auc_score(y_test, probs)
    print(f"ROC AUC: {auc:.4f}")
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0,1], [0,1], '--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/roc_curve.png')
    plt.close()

# ---------------------------
# Save pipeline
# ---------------------------
joblib.dump(pipeline, 'models/sleep_model.joblib')

# ---------------------------
# Feature importances
# ---------------------------
# Try to obtain feature names after preprocessing
try:
    pre = pipeline.named_steps['pre']
    # get_feature_names_out works in recent sklearn
    feat_names = list(pre.get_feature_names_out())
except Exception:
    # fallback: numeric names + OHE names from transformer
    numeric = numeric_features
    try:
        ohe = pre.transformers_[1][1].named_steps['ohe']
        cat_base = pre.transformers_[1][2]
        cat_names = list(ohe.get_feature_names_out(cat_base))
    except Exception:
        # a safe fallback naming if OHE method not available
        cat_names = [f"{c}_{v}" for c in cat_features for v in ['_unknown_']]
    feat_names = numeric + cat_names

# extract importances from RandomForest
clf = pipeline.named_steps['clf']
importances = clf.feature_importances_
# align lengths (in rare failure cases)
if len(importances) != len(feat_names):
    # try to trim or pad names
    min_len = min(len(importances), len(feat_names))
    feat_names = feat_names[:min_len]
    importances = importances[:min_len]

fi_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
fi_df.to_csv('results/feature_importances.csv', index=False)

# plot top features
plt.figure(figsize=(8, 5))
topn = min(12, len(fi_df))
plt.barh(fi_df['feature'].head(topn)[::-1], fi_df['importance'].head(topn)[::-1])
plt.xlabel('Importance')
plt.title('Top feature importances')
plt.tight_layout()
plt.savefig('results/feature_importances.png')
plt.close()

# ---------------------------
# Optional: SHAP explanations (if installed)
# ---------------------------
try:
    import shap
    # explain a sample of test data
    pre = pipeline.named_steps['pre']
    clf = pipeline.named_steps['clf']
    # transform a small subset to model space
    X_explain = X_test.sample(n=min(200, len(X_test)), random_state=42)
    X_trans = pre.transform(X_explain)
    # create TreeExplainer for RF
    explainer = shap.Explainer(clf, X_trans)
    shap_values = explainer(X_trans)
    # try to plot summary (save to file)
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('results/shap_summary.png')
    plt.close()
    print("Saved SHAP summary to results/shap_summary.png")
except Exception as e:
    print("SHAP not generated (shap may be missing or incompatible):", e)

# ---------------------------
# Save metrics + report
# ---------------------------
with open('results/metrics.txt', 'w') as f:
    f.write(f"accuracy: {acc:.4f}\n")
    f.write(f"precision: {prec:.4f}\n")
    f.write(f"recall: {rec:.4f}\n")
    f.write(f"f1: {f1:.4f}\n")
    if probs is not None:
        f.write(f"roc_auc: {auc:.4f}\n")
    f.write("\nclassification_report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nconfusion_matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

print("Saved model to models/sleep_model.joblib and metrics to results/metrics.txt")
print("Saved feature importances to results/feature_importances.csv and results/feature_importances.png")
