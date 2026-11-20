# How much of the test QWK comes from sharpness?

Ensemble on the test split: QWK 0.872 [0.859, 0.885] (n = 2240)

## 1. Sharpness alone (cut-points fitted on the train split)
QWK on test: 0.794 [0.776, 0.811]   (train fit: 0.792)
AUC for referable DR using sharpness alone: 0.794   (ensemble: 0.948)
Sharpness cut-points (grade 0|1|2|3|4): [193.8, 188.6, 57.6, 56.4]

## 2. Ensemble with the sharpness cue removed (re-weighted so grade is independent of sharpness)
Weighted QWK, ensemble: 0.747 [0.711, 0.777]   (effective n = 923)
Weighted QWK, sharpness alone: 0.421   (residual cue left after re-weighting; 0 = fully removed. Grades that barely overlap in sharpness, e.g. grade 0 vs 3/4, cannot be fully balanced)
Weighted AUC referable, ensemble: 0.918

Within sharpness terciles (unweighted):
        tercile   n sharpness_range                              grades  QWK_ensemble  AUC_referable
blurriest third 747           10-84    {1: 191, 2: 110, 3: 217, 4: 229}         0.839          0.965
   middle third 746          84-188      {1: 241, 2: 379, 3: 76, 4: 50}         0.620          0.871
 sharpest third 747        188-1246 {0: 572, 1: 76, 2: 83, 3: 13, 4: 3}         0.618          0.881

## 3. Within each true grade: does a blurrier image get a higher predicted grade?
 true_grade   n  median_sharpness  spearman(sharpness, predicted grade)     p
          0 572             244.8                                -0.269 6e-11
          1 508              95.1                                -0.498 3e-33
          2 572             110.3                                -0.337 1e-16
          3 306              52.1                                -0.704 5e-47
          4 282              43.2                                -0.318 5e-08

Negative correlations = the ensemble predicts higher grades for blurrier images of the SAME grade.
