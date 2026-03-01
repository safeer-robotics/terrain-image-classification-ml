from pathlib import Path
import cv2

DATASET_DIR = Path(".")  # because your class folders are in the root

classes = ["Grassy_Terrain", "Sandy_Terrain", "Rocky_Terrain", "Marshy_Terrain", "Other_Image"]

print("Checking folders and image counts...\n")

total = 0
for c in classes:
    folder = DATASET_DIR / c
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    imgs = list(folder.glob("*.jpg"))
    print(f"{c:15s} -> {len(imgs)} images")
    total += len(imgs)

print(f"\nTotal images found: {total}")

# Load 1 sample image
sample = next((DATASET_DIR / classes[0]).glob("*.jpg"))
img = cv2.imread(str(sample))
if img is None:
    raise ValueError(f"OpenCV failed to read image: {sample}")

print(f"\nSample image loaded: {sample.name}")
print("Image shape (H, W, C):", img.shape)
