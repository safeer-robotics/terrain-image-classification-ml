import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

print("Step 1: Loading features...")
X = np.load("X_features.npy")
y = np.load("y_labels.npy")
print("Loaded X:", X.shape, " y:", y.shape)

print("\nStep 2: Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Train:", X_train.shape, " Test:", X_test.shape)

print("\nStep 3: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling done.")

print("\nStep 4: Training SVM (this may take some time)...")
svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train_scaled, y_train)
print("SVM training done.")

print("\nStep 5: Predicting...")
y_pred = svm.predict(X_test_scaled)
print("Prediction done.")

print("\nStep 6: Evaluating...")
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("\nSVM Results")
print("Accuracy:", acc)
print("F1-score (macro):", f1)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nStep 7: Saving model + scaler...")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(svm, "svm_model.pkl")
print("Saved: scaler.pkl and svm_model.pkl")
