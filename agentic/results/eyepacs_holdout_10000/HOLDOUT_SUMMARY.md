# Held-out EyePACS evaluation: top 10000 sharpest unused images

## A_summary
                             value
images                       10000
patients                      6401
grades                  {0: 10000}
fellow_eye_available_%        72.0
accuracy                    0.1696
QWK                            0.0
AUC_referable                 None
sens_referable_%               0.0
spec_referable_%              74.1

## B_policies
                                      policy  auto_cleared  missed_referable  miss_rate_%  missed_per_1000  clear_rate_%  referral_recall_%  overrides  overrides_referable  severe_or_PDR_cleared
0  top-grade gate (max p >= 0.85) [baseline]           871                 0          0.0              0.0           8.7                0.0          0                    0                      0
1                 binary gate (P_nr >= 0.94)           612                 0          0.0              0.0           6.1                0.0          0                    0                      0
2                      binary + lesion check           598                 0          0.0              0.0           6.0                0.0         14                    0                      0
3                        binary + fellow eye           587                 0          0.0              0.0           5.9                0.0         25                    0                      0
4  binary + lesion + fellow eye (fixed rule)           573                 0          0.0              0.0           5.7                0.0         39                    0                      0
5  threshold only, matched clears (t=0.9421)           572                 0          0.0              0.0           5.7                0.0          0                    0                      0

## C_paired
                                                 comparison  misses_first  misses_second  missed_only_by_first  missed_only_by_second  misses_avoided avoided_95CI  exact_McNemar_p
0  top-grade gate vs verify-before-clear (thesis: p = 0.11)             0              0                     0                      0               0       [0, 0]              1.0
1                        binary gate vs verify-before-clear             0              0                     0                      0               0       [0, 0]              1.0
2                  matched threshold vs verify-before-clear             0              0                     0                      0               0       [0, 0]              1.0

## D_threshold_sweep
      t threshold_only (clears/missed) verify (clears/missed) threshold_only_matched (clears/missed)
0  0.90                         1298/0                 1215/0                                 1214/0
1  0.91                         1114/0                 1036/0                                 1035/0
2  0.92                          925/0                  866/0                                  865/0
3  0.93                          757/0                  709/0                                  708/0
4  0.94                          612/0                  573/0                                  572/0
5  0.95                          450/0                  424/0                                  423/0
6  0.96                          329/0                  311/0                                  310/0
7  0.97                          218/0                  203/0                                  202/0
