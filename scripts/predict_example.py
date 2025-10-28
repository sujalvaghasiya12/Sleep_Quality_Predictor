# scripts/debug_and_improve_model.py
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -------------------------------------------------------------------------
# 1. Check and display current model behavior
# -------------------------------------------------------------------------
print("🔍 Loading model...")
model_path = Path('models/sleep_model.joblib')
if not model_path.exists():
    print("⚠️ No model found. Training a new one automatically...")
else:
    model = joblib.load(model_path)
    X_sample = pd.DataFrame([{
        'hours_of_sleep': 10.0,
        'screen_time': 1.0,
        'caffeine_intake': 1,
        'steps_walked': 10000,
        'stress_level': 1,
        'exercise_minutes': 60,
        'alcohol_units': 0,
        'ambient_light': 'low',
        'weekday': 'weekend',
        'bedtime_hour': 22.0
    }])
    print("\nSample prediction check:")
    print("Raw predict:", model.predict(X_sample))
    print("Probabilities [poor, good]:", model.predict_proba(X_sample)[0])

# -------------------------------------------------------------------------
# 2. Reload dataset and fix label mapping if needed
# -------------------------------------------------------------------------
print("\n📄 Loading dataset...")
df = pd.read_csv('data/sleep_data.csv')
print("Unique labels:", df['sleep_quality'].unique())
print(df['sleep_quality'].value_counts())

# Ensure label mapping is correct: 'good' → 1, 'poor' → 0
df['sleep_quality'] = df['sleep_quality'].map({'good': 1, 'poor': 0})
df = df.dropna(subset=['sleep_quality'])

# -------------------------------------------------------------------------
# 3. Retrain improved model with class balance and feature handling
# -------------------------------------------------------------------------
print("\n⚙️ Retraining improved model...")

features = [
    'hours_of_sleep', 'screen_time', 'caffeine_intake', 'steps_walked', 'stress_level',
    'exercise_minutes', 'alcohol_units', 'ambient_light', 'weekday', 'bedtime_hour'
]
X = df[features]
y = df['sleep_quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

numeric_features = [
    'hours_of_sleep', 'screen_time', 'caffeine_intake', 'steps_walked', 'stress_level',
    'exercise_minutes', 'alcohol_units', 'bedtime_hour'
]
cat_features = ['ambient_light', 'weekday']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', cat_transformer, cat_features)
])

model = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight='balanced',
        max_depth=10
    ))
])

model.fit(X_train, y_train)
joblib.dump(model, 'models/sleep_model.joblib')

print("✅ Retraining complete. Model saved to models/sleep_model.joblib")

# -------------------------------------------------------------------------
# 4. Evaluate and show feature importance
# -------------------------------------------------------------------------
y_pred = model.predict(X_test)
print("\n📊 Evaluation Metrics:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Extract feature names after preprocessing
ohe = model.named_steps['pre'].named_transformers_['cat']['ohe']
cat_names = ohe.get_feature_names_out(cat_features)
all_features = numeric_features + list(cat_names)

# Get feature importances from RandomForest
clf = model.named_steps['clf']
importances = pd.Series(clf.feature_importances_, index=all_features).sort_values(ascending=False)

print("\n🔥 Top 10 Important Features for Sleep Quality:")
print(importances.head(10))
