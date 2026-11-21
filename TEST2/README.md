# TEST2 — ensemble retrained on a randomly sampled EyePACS subset

The thesis 15k subset was selected sharpest-first within each grade, which made image sharpness fall
with grade (median 245 for grade 0 vs 42–49 for grades 3–4) and let the models use sharpness as a
shortcut. TEST2 samples 15,000 images **at random** within each grade (5000 / 2250 / 4500 / 1750 / 1500
for grades 0–4) and retrains ResNet-50 and ViT-B/16 with the unchanged thesis recipe.

Run from the repo root:

    python TEST2/run_test2.py

- Large files (crops, checkpoints, manifests) go to `E:\TEST2`; small results to `TEST2/results/`.
- Every step is cached and resumable; re-running continues where it stopped.
- Existing 512×512 crops in `REUSE_DIRS` are reused instead of being recreated.
- Results: `TEST2/results/TEST2_SUMMARY.md`.
