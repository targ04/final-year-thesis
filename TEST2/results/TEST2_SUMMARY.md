# TEST2: ensemble retrained on a randomly sampled 15k EyePACS subset

Thesis 15k (sharpness-selected), test: ensemble accuracy 0.727, QWK 0.872; referable sens 86.6%, spec 88.1%, AUC 0.948; top-grade gate 396 cleared / 16 missed; binary 0.94 gate 423 / 13.

Ensemble weight (val-selected): w_ResNet = 0.60; temperature T = 0.9308 (fitted on val)

## Grading (patient-level bootstrap 95% CI)
split     model             accuracy                  QWK
  val ResNet-50 0.564 [0.542, 0.586] 0.715 [0.688, 0.742]
  val  ViT-B/16 0.521 [0.498, 0.542] 0.680 [0.651, 0.707]
  val  Ensemble 0.582 [0.560, 0.604] 0.733 [0.707, 0.758]
 test ResNet-50 0.553 [0.531, 0.574] 0.733 [0.706, 0.756]
 test  ViT-B/16 0.491 [0.469, 0.514] 0.680 [0.650, 0.707]
 test  Ensemble 0.561 [0.540, 0.583] 0.740 [0.713, 0.764]

## Test confusion matrix (ensemble)
        pred 0  pred 1  pred 2  pred 3  pred 4
true 0     572     105      62       2       7
true 1     227      83      45       1       7
true 2     138      58     287     103      37
true 3      13       4      68     167      16
true 4       7       1      25      54     143

## Per-grade (test)
 grade  precision  recall    F1   n
     0      0.598   0.765 0.671 748
     1      0.331   0.229 0.270 363
     2      0.589   0.461 0.517 623
     3      0.511   0.623 0.561 268
     4      0.681   0.622 0.650 230

## Binary readouts (argmax operating point; AUC of P(grade >= k))
                       outcome split  prevalence_%  sensitivity_%  specificity_%  PPV_%  NPV_%    AUC
         Referable, grade >= 2   val          50.6           76.9           90.4   89.1   79.2 0.9093
         Referable, grade >= 2  test          50.2           80.3           88.8   87.9   81.7 0.9161
Vision-threatening, grade >= 3   val          20.1           81.0           92.0   71.7   95.1 0.9420
Vision-threatening, grade >= 3  test          22.3           76.3           90.9   70.8   93.0 0.9331

## Calibration (test)
probabilities  ECE_5class  ECE_binary_P_nr
 uncalibrated      0.0295           0.0303
    T = 0.931      0.0267           0.0217

Binary-gate threshold re-selected on val (lowest t with val misses <= top-grade val misses = 0): t* = 0.985

## Triage gates (no verification)
split                              policy  auto_cleared  missed_referable  miss_rate_%  referral_recall_%  severe_or_PDR_cleared  workload_%
  val            Top-grade: max p >= 0.85             4                 0         0.00             100.00                      0       99.82
  val       P_nr >= 0.85 (same threshold)           572                50         8.74              95.57                      2       74.36
  val          Binary: P_nr >= t* = 0.985             2                 0         0.00             100.00                      0       99.91
  val Binary: P_nr >= 0.94 (thesis value)           130                 7         5.38              99.38                      0       94.17
 test            Top-grade: max p >= 0.85             8                 0         0.00             100.00                      0       99.64
 test       P_nr >= 0.85 (same threshold)           552                51         9.24              95.45                      4       75.27
 test          Binary: P_nr >= t* = 0.985             2                 0         0.00             100.00                      0       99.91
 test Binary: P_nr >= 0.94 (thesis value)           128                 9         7.03              99.20                      1       94.27

## Image sharpness by grade (val + test images)
   TEST2 median sharpness  thesis 15k median (for reference)
0                    56.4                              244.9
1                    64.7                               95.0
2                    47.8                              111.6
3                    47.9                               49.3
4                    42.9                               41.9
