# APTOS 2019 external evaluation — all (n = 3662)
Grade counts: {0: 1805, 1: 370, 2: 999, 3: 193, 4: 295}

## A_grading
      model  accuracy     QWK          QWK_CI
0    resnet    0.1450  0.4370  [0.417, 0.457]
1       vit    0.1275  0.3936  [0.375, 0.413]
2  ensemble    0.1393  0.4467  [0.427, 0.466]

## A_confusion
        pred 0  pred 1  pred 2  pred 3  pred 4
true 0       7    1151     250     364      33
true 1       0       6     198     137      29
true 2       0       0     177     772      50
true 3       0       0      10     143      40
true 4       0       0      17     101     177

## B_calibration
  probabilities  ECE_5class  ECE_binary_P(nonref)
0  uncalibrated      0.5812                0.3286
1      T=1.2308      0.5386                0.3377

## C_binary
   prevalence_referable_%  sensitivity_%  specificity_%     AUC          AUC_CI
0                    40.6          100.0           53.5  0.9633  [0.958, 0.969]

## D_triage_counts
                           route  images
0  severity referral (pred >= 2)    2498
1        low-confidence referral    1147
2           auto-clear candidate      17

## D_policies
                                      policy  auto_cleared  missed_referable  miss_rate_%  referral_recall_%  clear_rate_%  overrides  overrides_referable  severe_or_PDR_cleared
0                   binary gate P_nr >= 0.94            17                 0          0.0              100.0           0.5          0                    0                      0
1                 binary gate + lesion check            16                 0          0.0              100.0           0.4          1                    0                      0
2  threshold only, matched clears (t=0.9419)            15                 0          0.0              100.0           0.4          0                    0                      0

## D_paired
                         comparison  missed_only_by_first  missed_only_by_second  exact_McNemar_p
0               gate vs gate+lesion                     0                      0              1.0
1  matched threshold vs gate+lesion                     0                      0              1.0

## D_missed_grades
Empty DataFrame
Columns: [gate, gate+lesion]
Index: []

## E_threshold_sweep
      t  threshold_only_clears  threshold_only_missed  verify_clears  verify_missed  matched_threshold_clears  matched_threshold_missed
0  0.90                     56                      0             55              0                        54                         0
1  0.91                     44                      0             43              0                        42                         0
2  0.92                     38                      0             37              0                        36                         0
3  0.93                     26                      0             25              0                        24                         0
4  0.94                     17                      0             16              0                        15                         0
5  0.95                      9                      0              8              0                         7                         0
6  0.96                      6                      0              5              0                         4                         0
7  0.97                      2                      0              2              0                         1                         0

## F_lesion_vs_grade
    feature  spearman_vs_grade  AUC_grade>=2  AUC_grade>=3  median_grade0  median_grade4
0  MA_count              0.808         0.936         0.822            2.0           39.0
1  HE_count              0.741         0.903         0.850            2.0           27.0
2  EX_count              0.732         0.913         0.795            2.0           38.0
3  SE_count              0.584         0.808         0.794            0.0            3.0


# APTOS 2019 external evaluation — crop_valid (n = 3537)
Grade counts: {0: 1716, 1: 360, 2: 983, 3: 187, 4: 291}

## A_grading
      model  accuracy     QWK          QWK_CI
0    resnet    0.1484  0.4426  [0.423, 0.463]
1       vit    0.1298  0.4038  [0.385, 0.423]
2  ensemble    0.1422  0.4558  [0.436, 0.476]

## A_confusion
        pred 0  pred 1  pred 2  pred 3  pred 4
true 0       7    1100     249     329      31
true 1       0       6     197     131      26
true 2       0       0     176     758      49
true 3       0       0      10     139      38
true 4       0       0      17      99     175

## B_calibration
  probabilities  ECE_5class  ECE_binary_P(nonref)
0  uncalibrated      0.5780                0.3260
1      T=1.2308      0.5354                0.3351

## C_binary
   prevalence_referable_%  sensitivity_%  specificity_%    AUC          AUC_CI
0                    41.3          100.0           53.6  0.963  [0.958, 0.968]

## D_triage_counts
                           route  images
0  severity referral (pred >= 2)    2424
1        low-confidence referral    1101
2           auto-clear candidate      12

## D_policies
                                      policy  auto_cleared  missed_referable  miss_rate_%  referral_recall_%  clear_rate_%  overrides  overrides_referable  severe_or_PDR_cleared
0                   binary gate P_nr >= 0.94            12                 0          0.0              100.0           0.3          0                    0                      0
1                 binary gate + lesion check            12                 0          0.0              100.0           0.3          0                    0                      0
2  threshold only, matched clears (t=0.9413)            11                 0          0.0              100.0           0.3          0                    0                      0

## D_paired
                         comparison  missed_only_by_first  missed_only_by_second  exact_McNemar_p
0               gate vs gate+lesion                     0                      0              1.0
1  matched threshold vs gate+lesion                     0                      0              1.0

## D_missed_grades
Empty DataFrame
Columns: [gate, gate+lesion]
Index: []

## E_threshold_sweep
      t  threshold_only_clears  threshold_only_missed  verify_clears  verify_missed  matched_threshold_clears  matched_threshold_missed
0  0.90                     48                      0             48              0                        47                         0
1  0.91                     37                      0             37              0                        36                         0
2  0.92                     31                      0             31              0                        30                         0
3  0.93                     19                      0             19              0                        18                         0
4  0.94                     12                      0             12              0                        11                         0
5  0.95                      6                      0              6              0                         5                         0
6  0.96                      5                      0              5              0                         4                         0
7  0.97                      2                      0              2              0                         1                         0

## F_lesion_vs_grade
    feature  spearman_vs_grade  AUC_grade>=2  AUC_grade>=3  median_grade0  median_grade4
0  MA_count              0.808         0.936         0.817            2.0           38.0
1  HE_count              0.741         0.902         0.846            2.0           27.0
2  EX_count              0.735         0.914         0.794            2.0           38.0
3  SE_count              0.589         0.810         0.792            0.0            3.0
