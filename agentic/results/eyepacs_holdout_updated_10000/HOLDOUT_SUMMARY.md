# Held-out EyePACS evaluation: 10000 unused images, sampling = balanced (equal grades [0, 1, 2], sharpest first)

## A_summary
                                              value
images                                         9222
patients                                       6632
grades                  {0: 3333, 1: 2556, 2: 3333}
fellow_eye_available_%                         98.8
accuracy                                     0.3121
QWK                                          0.3845
AUC_referable                                0.7283
sens_referable_%                               65.6
spec_referable_%                               60.8

## B_policies
                                      policy  auto_cleared  missed_referable  miss_rate_%  missed_per_1000  clear_rate_%  referral_recall_%  overrides  overrides_referable  severe_or_PDR_cleared
0  top-grade gate (max p >= 0.85) [baseline]           666               162        24.32            17.57           7.2              95.14          0                    0                      0
1                 binary gate (P_nr >= 0.94)           509                35         6.88             3.80           5.5              98.95          0                    0                      0
2                      binary + lesion check           500                34         6.80             3.69           5.4              98.98          9                    1                      0
3                        binary + fellow eye           465                31         6.67             3.36           5.0              99.07         44                    4                      0
4  binary + lesion + fellow eye (fixed rule)           456                30         6.58             3.25           4.9              99.10         53                    5                      0
5  threshold only, matched clears (t=0.9444)           455                31         6.81             3.36           4.9              99.07          0                    0                      0

## C_paired
                                                 comparison  misses_first  misses_second  missed_only_by_first  missed_only_by_second  misses_avoided avoided_95CI  exact_McNemar_p
0  top-grade gate vs verify-before-clear (thesis: p = 0.11)           162             30                   132                      0             132   [110, 155]           0.0000
1                        binary gate vs verify-before-clear            35             30                     5                      0               5      [1, 10]           0.0625
2                  matched threshold vs verify-before-clear            31             30                     4                      3               1      [-4, 6]           1.0000

## D_threshold_sweep
      t threshold_only (clears/missed) verify (clears/missed) threshold_only_matched (clears/missed)
0  0.90                        1014/96                 888/76                                 887/81
1  0.91                         888/81                 776/67                                 775/68
2  0.92                         749/64                 664/55                                 663/54
3  0.93                         633/50                 560/42                                 559/41
4  0.94                         509/35                 456/30                                 455/31
5  0.95                         390/26                 354/23                                 353/23
6  0.96                         278/18                 251/15                                 250/15
7  0.97                          185/8                  167/7                                  166/7
