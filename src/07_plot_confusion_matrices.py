import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import joblib

# Class names in label order
class_names = ["Grassy", "Sandy", "Rocky", "Marshy", "Other"]

# Load saved features
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# Recreate the same split (must match earlier scripts)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load models
svm = joblib.load("svm_model.pkl")
rf = joblib.load("rf_model.pkl")
knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Predictions
X_test_scaled = scaler.transform(X_test)
svm_pred = svm.predict(X_test_scaled)
knn_pred = knn.predict(X_test_scaled)
rf_pred = rf.predict(X_test)

def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure()
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

plot_cm(y_test, svm_pred, "Confusion Matrix - SVM", "cm_svm.png")
plot_cm(y_test, rf_pred, "Confusion Matrix - Random Forest", "cm_rf.png")
plot_cm(y_test, knn_pred, "Confusion Matrix - KNN", "cm_knn.png")

print("Saved: cm_svm.png, cm_rf.png, cm_knn.png")
