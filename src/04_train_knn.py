import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

# Load features
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("Step 1: Loaded features")
print("X:", X.shape, " y:", y.shape)

# Same split for fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nStep 2: Split done")
print("Train:", X_train.shape, " Test:", X_test.shape)

# Scale features (required for KNN)
print("\nStep 3: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling done.")

# Train KNN
print("\nStep 4: Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(X_train_scaled, y_train)
print("KNN training done.")

# Predict
print("\nStep 5: Predicting...")
y_pred = knn.predict(X_test_scaled)
print("Prediction done.")

# Evaluate
print("\nStep 6: Evaluating...")
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("\nKNN Results")
print("Accuracy:", acc)
print("F1-score (macro):", f1)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(knn, "knn_model.pkl")
print("\nSaved: knn_model.pkl")
