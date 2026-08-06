"""
Fundus image preprocessing: contour-based circular crop + resize to 512x512.

Crops out the black background around the retinal fundus circle using
contour detection, then resizes the cropped region to a fixed 512x512.

Usage:
    1. Set SOURCE_PATH and DEST_PATH below.
    2. Set TEST = True to preview 5 sample crops before running on everything.
    3. Set TEST = False to process the entire folder.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
SOURCE_PATH = r"D:\kagglehub\datasets\mohlamin\resized-eyepacs-diabetic-retinopathy-dataset\versions\1\Images"      # <-- set your input folder
DEST_PATH = r"D:\Thesis Dataset\cropped and resized"           # <-- set your output folder
TARGET_SIZE = 512
TEST = False                                        # True = preview only, False = process all
NUM_TEST_SAMPLES = 5
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
# -----------------------------------------


def find_fundus_bbox(image, threshold=10):
    """
    Detects the circular fundus region using contour detection on a
    grayscale-thresholded mask, and returns the bounding box (x, y, w, h)
    of the largest contour. Falls back to the full image if nothing found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary mask: anything brighter than `threshold` is considered fundus
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Clean up small noise specks so they don't get picked as contours
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = image.shape[:2]
        return 0, 0, w, h

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def crop_and_resize(image, target_size=TARGET_SIZE):
    """Crops the fundus region and resizes it to target_size x target_size."""
    x, y, w, h = find_fundus_bbox(image)

    # Guard against degenerate boxes (e.g. fully black images)
    if w < 10 or h < 10:
        cropped = image
    else:
        cropped = image[y:y + h, x:x + w]

    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def list_images(folder):
    files = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(VALID_EXTENSIONS):
            files.append(fname)
    return files


def run_test_preview(source_path, num_samples=NUM_TEST_SAMPLES):
    files = list_images(source_path)[:num_samples]
    if not files:
        print(f"No images found in {source_path}")
        return

    fig, axes = plt.subplots(len(files), 2, figsize=(6, 3 * len(files)))
    if len(files) == 1:
        axes = [axes]

    for i, fname in enumerate(files):
        img_path = os.path.join(source_path, fname)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read {fname}, skipping.")
            continue

        processed = crop_and_resize(image)

        orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        proc_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        axes[i][0].imshow(orig_rgb)
        axes[i][0].set_title(f"Original: {fname}")
        axes[i][0].axis("off")

        axes[i][1].imshow(proc_rgb)
        axes[i][1].set_title("Cropped + Resized (512x512)")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()
    print(f"Previewed {len(files)} sample image(s). Set TEST = False to process the full folder.")


def run_full_processing(source_path, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    files = list_images(source_path)

    if not files:
        print(f"No images found in {source_path}")
        return

    total = len(files)
    for i, fname in enumerate(files, start=1):
        img_path = os.path.join(source_path, fname)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read {fname}, skipping.")
            continue

        processed = crop_and_resize(image)

        out_path = os.path.join(dest_path, fname)
        cv2.imwrite(out_path, processed)

        if i % 100 == 0 or i == total:
            print(f"Processed {i}/{total} images")

    print(f"Done. {total} images saved to {dest_path}")


if __name__ == "__main__":
    if TEST:
        run_test_preview(SOURCE_PATH, NUM_TEST_SAMPLES)
    else:
        run_full_processing(SOURCE_PATH, DEST_PATH)