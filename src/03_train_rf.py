import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

# Load features
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("Step 1: Loaded features")
print("X:", X.shape, " y:", y.shape)

# Same split settings for fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nStep 2: Split done")
print("Train:", X_train.shape, " Test:", X_test.shape)

print("\nStep 3: Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("RF training done.")

print("\nStep 4: Predicting...")
y_pred = rf.predict(X_test)
print("Prediction done.")

print("\nStep 5: Evaluating...")
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("\nRandom Forest Results")
print("Accuracy:", acc)
print("F1-score (macro):", f1)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nStep 6: Saving model...")
joblib.dump(rf, "rf_model.pkl")
print("Saved: rf_model.pkl")
