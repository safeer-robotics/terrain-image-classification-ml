from pathlib import Path
import numpy as np
import cv2
from skimage.feature import hog
from tqdm import tqdm

# Your class folders (exact names)
CLASSES = ["Grassy_Terrain", "Sandy_Terrain", "Rocky_Terrain", "Marshy_Terrain", "Other_Image"]

# Because your folders are at project root:
DATASET_DIR = Path(".")

# Resize for speed + consistency
IMG_SIZE = (128, 128)  # (width, height)

def extract_features(img_bgr):
    """Return a single 1D feature vector using HOG + HSV color histogram."""
    # Resize
    img_bgr = cv2.resize(img_bgr, IMG_SIZE)

    # ---- HOG (on grayscale) ----
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    # ---- Color histogram (HSV) ----
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Combine
    return np.hstack([hog_feat, hist]).astype(np.float32)

def main():
    X = []
    y = []
    bad_files = 0

    for label, cls in enumerate(CLASSES):
        folder = DATASET_DIR / cls
        images = list(folder.glob("*.jpg"))

        for img_path in tqdm(images, desc=f"Processing {cls}"):
            img = cv2.imread(str(img_path))
            if img is None:
                bad_files += 1
                continue
            feat = extract_features(img)
            X.append(feat)
            y.append(label)

    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)

    print("\nDone!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Bad/unreadable files:", bad_files)

    # Save for training step
    np.save("X_features.npy", X)
    np.save("y_labels.npy", y)
    print("\nSaved: X_features.npy and y_labels.npy")

if __name__ == "__main__":
    main()
