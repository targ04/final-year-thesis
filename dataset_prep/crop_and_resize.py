"""
Fundus image preprocessing: contour-based circular crop + resize to 512x512.

Crops out the black background around the retinal fundus circle using
contour detection, then resizes the cropped region to a fixed 512x512.

Also flags structurally invalid images (e.g. composite/corrupted files
containing multiple separate fundus blobs merged into one file) using two
geometric checks on the detected contour:
    1. Contour count  - more than one significant bright blob suggests a
       composite/corrupted image rather than a single fundus photo.
    2. Circularity     - a real fundus photo's bright region should be
       roughly circular/elliptical. A bounding box that spans multiple
       separate blobs will have low circularity even if the box itself
       looks roughly square.

Flagged images are NOT written to DEST_PATH. Instead they are logged to
a rejects CSV (REJECTS_CSV_PATH) with the filename and rejection reason,
so they never make it into the folder that later scripts (e.g. sharpness
scoring, selection) will scan.

Usage:
    1. Set SOURCE_PATH and DEST_PATH below.
    2. Set TEST = True to preview 5 sample crops (with validity shown)
       before running on everything.
    3. Set TEST = False to process the entire folder.
"""

import os
import csv
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
SOURCE_PATH = r"D:\kagglehub\datasets\mohlamin\resized-eyepacs-diabetic-retinopathy-dataset\versions\1\Images"      # <-- set your input folder
DEST_PATH = r"D:\Thesis Dataset\cropped and resized new"           # <-- set your output folder
REJECTS_CSV_PATH = r"D:\Thesis Dataset\labels\crop_rejects.csv"  # <-- set your rejects log path
TARGET_SIZE = 512
TEST = False                                        # True = preview only, False = process all
NUM_TEST_SAMPLES = 5
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# A contour is "significant" if its area exceeds this fraction of the
# total image area. Small specks that survived the morphological cleanup
# shouldn't count toward the multi-contour check.
MIN_CONTOUR_AREA_FRACTION = 0.01

# Circularity = 4*pi*area / perimeter^2. A perfect circle scores 1.0.
# Below this threshold, the largest contour is considered too irregular
# to be a single clean fundus capture.
CIRCULARITY_THRESHOLD = 0.6
# -----------------------------------------


def analyze_fundus_mask(image, threshold=10):
    """
    Detects the fundus region(s) using contour detection on a
    grayscale-thresholded mask, and returns:
        (x, y, w, h, is_valid, reason)
    where (x, y, w, h) is the bounding box of the largest contour,
    is_valid indicates whether the image passes the geometric sanity
    checks, and reason explains a failure (or "ok" if valid).
    Falls back to the full image (marked valid) if no contours are found,
    since that typically means an all-black/degenerate image that gets
    caught separately by the small-box guard in crop_and_resize.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    image_area = img_h * img_w

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, 0, img_w, img_h, True, "no_contours_found"

    significant = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA_FRACTION * image_area]

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    if len(significant) > 1:
        return x, y, w, h, False, f"multiple_contours ({len(significant)})"

    if circularity < CIRCULARITY_THRESHOLD:
        return x, y, w, h, False, f"low_circularity ({circularity:.2f})"

    return x, y, w, h, True, "ok"


def crop_and_resize(image, target_size=TARGET_SIZE):
    """
    Crops the fundus region and resizes it to target_size x target_size.
    Returns (resized_image, is_valid, reason).
    """
    x, y, w, h, is_valid, reason = analyze_fundus_mask(image)

    # Guard against degenerate boxes (e.g. fully black images)
    if w < 10 or h < 10:
        cropped = image
    else:
        cropped = image[y:y + h, x:x + w]

    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized, is_valid, reason


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

        processed, is_valid, reason = crop_and_resize(image)

        orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        proc_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        axes[i][0].imshow(orig_rgb)
        axes[i][0].set_title(f"Original: {fname}")
        axes[i][0].axis("off")

        status = "VALID" if is_valid else f"REJECTED: {reason}"
        axes[i][1].imshow(proc_rgb)
        axes[i][1].set_title(f"Cropped + Resized\n{status}")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()
    print(f"Previewed {len(files)} sample image(s). Set TEST = False to process the full folder.")


def run_full_processing(source_path, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    os.makedirs(os.path.dirname(REJECTS_CSV_PATH), exist_ok=True)
    files = list_images(source_path)

    if not files:
        print(f"No images found in {source_path}")
        return

    total = len(files)
    saved = 0
    rejected = 0
    rejects_log = []

    for i, fname in enumerate(files, start=1):
        img_path = os.path.join(source_path, fname)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read {fname}, skipping.")
            rejects_log.append((fname, "unreadable"))
            rejected += 1
            continue

        processed, is_valid, reason = crop_and_resize(image)

        if not is_valid:
            rejects_log.append((fname, reason))
            rejected += 1
        else:
            out_path = os.path.join(dest_path, fname)
            cv2.imwrite(out_path, processed)
            saved += 1

        if i % 100 == 0 or i == total:
            print(f"Processed {i}/{total} images ({saved} saved, {rejected} rejected)")

    with open(REJECTS_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "reason"])
        writer.writerows(rejects_log)

    print(f"\nDone. {saved} images saved to {dest_path}")
    print(f"{rejected} images rejected and logged to {REJECTS_CSV_PATH}")


if __name__ == "__main__":
    if TEST:
        run_test_preview(SOURCE_PATH, NUM_TEST_SAMPLES)
    else:
        run_full_processing(SOURCE_PATH, DEST_PATH)