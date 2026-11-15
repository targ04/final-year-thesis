# Report notes: binary referral gate, comparison with LumineticsCore, verify-before-clear

Every number here can be regenerated:
`python agentic/binary_gate_analysis.py` → `python agentic/offline_ablation.py`.
Test set: 2,240 images. Val set: 2,255 images. All thresholds were selected on val; test was used once.

## 1. Framing for the report

The clinical decision is **binary**: refer at moderate NPDR or worse (grade ≥ 2). This is the
"more-than-mild DR" (mtmDR) threshold used by FDA-cleared autonomous systems such as LumineticsCore.
The midsem auto-clear gate instead thresholded the **probability of the single most likely grade**
(≥ 0.85). As a result, a case with P(grade 0) = 0.75 and P(grade 1) = 0.20 was referred for "low
confidence", even though the model was 95% confident the patient was **non-referable**.

The v2 gate measures confidence in the decision actually being made:
**auto-clear a grade 0/1 prediction only if P(grade 0) + P(grade 1) ≥ t.**
t is selected on val as the lowest threshold whose val misses don't exceed the v1 gate's
(same safety, as many clears as possible). This gives **t = 0.94**.

## 2. The 5-class model read as a binary detector

Rule: argmax grade ≥ 2 = referable. Output: `binary_operating_point.csv`.

| Split | Referable prevalence | Sensitivity | Specificity | AUC of P(grade ≥ 2) |
|---|---|---|---|---|
| Val | 51.66% | 86.09% | 88.17% | 0.9475 |
| Test | 51.79% | 86.55% | 88.15% | 0.9480 |

For reference, the **LumineticsCore (IDx-DR) pivotal trial** reported sensitivity 87.2%
(95% CI 81.8–91.2%), specificity 90.7% (95% CI 88.3–92.7%) and imageability 96.1%
(Abràmoff et al., *npj Digital Medicine* 1:39, 2018, doi:10.1038/s41746-018-0040-6).
Real-world adopters report sensitivity of 87–100% and specificity of 60–91% (Teng et al. [13]).

**Comparable order of magnitude, not directly comparable.** State the differences:
- **Reference standard:** the trial used Reading Center grading of widefield stereo photographs plus OCT; this project uses single-image EyePACS labels.
- **Unit of decision:** the trial decided per participant (either eye, DME included); this project decides per image.
- **Prevalence:** 23.8% mtmDR in the trial, against a deliberately enriched 51.8% here.
- **Population and device:** the trial ran prospectively in primary care with a single camera system.

This also answers "commercial systems reach ~0.9 but you get ~0.73". The 0.73 is **5-class accuracy**.
On the like-for-like binary task the model reaches about 87% sensitivity and 88% specificity, with an AUC of 0.948.
Figure: `binary_roc.png`. It shows the ROC for val and test, the model's operating point, and the
published pivotal-trial point, marked "different data".

## 3. v1 vs v2 auto-clear gate

Output: `binary_gate_comparison.csv`.

| Split | Gate | Auto-cleared | Missed grade ≥ 2 | Miss rate | Referral recall | Workload |
|---|---|---|---|---|---|---|
| Val | v1: top-grade probability ≥ 0.85 | 413 | 11 | 2.66% | 99.06% | 81.69% |
| Val | P(0+1) ≥ 0.85 (same threshold) | 641 | 30 | 4.68% | 97.42% | 71.57% |
| Val | **v2: P(0+1) ≥ 0.94** | 410 | 10 | 2.44% | 99.14% | 81.82% |
| Test | v1: top-grade probability ≥ 0.85 | 396 | 16 | 4.04% | 98.62% | 82.32% |
| Test | P(0+1) ≥ 0.85 (same threshold) | 631 | 31 | 4.91% | 97.33% | 71.83% |
| Test | **v2: P(0+1) ≥ 0.94** | **423** | **13** | **3.07%** | **98.88%** | **81.12%** |

- On test, v2 is better on both axes: 27 more patients auto-cleared and 3 fewer missed referable cases.
- Swapping to the summed probability but keeping 0.85 is **unsafe**: it misses twice as many cases. The summed probability is always at least the top-grade probability, so the threshold has to be re-selected.
- Of the test cases v1 referred for low confidence, v2 now clears 109. Their true grades are 86 grade 0, 19 grade 1, 4 grade 2 and 0 grade 3/4. Net: +27 clears, −3 misses.

## 4. Verify-before-clear on top of the v2 gate

Output: `ablation_results.csv`. Lesion rule chosen on val: MA_count > 11 (95th percentile of
correctly cleared val cases, overriding at most 10% of clears).

| Test policy | Auto-cleared | Missed | Miss rate | Recall | Overrides |
|---|---|---|---|---|---|
| P0-v1 (midsem: v1 gate, no verification) | 396 | 16 | 4.04% | 98.62% | 0 |
| P0 (v2 gate, no verification) | 423 | 13 | 3.07% | 98.88% | 0 |
| P1: tighter gate threshold, same workload as P3 | 404 | 13 | 3.22% | 98.88% | 0 |
| P2: + fellow eye | 412 | 10 | 2.43% | 99.14% | 11 |
| P3: + lesion check | 405 | 12 | 2.96% | 98.97% | 18 |
| P1b: tighter gate threshold, same workload as P4 | 395 | 13 | 3.29% | 98.88% | 0 |
| **P4: + lesion + fellow eye (fixed rule)** | **396** | **10** | **2.53%** | **99.14%** | 27 |

**Headline:** compared with the midsem policy, the full v2 policy clears exactly as many patients (396)
but misses **10 instead of 16** referable cases (−37.5%). At the same workload, simply tightening the
threshold misses 13, so the verification adds safety that thresholding alone can't.
Figure: `safety_workload_tradeoff.png`. On test, the verify-before-clear curve stays below the
threshold-only curve across the whole range.

**Caveats to state:**
- The counts are small (10–16 misses), so differences are a few cases.
- The fellow eye is only available when both eyes are in the same split.
- TTA and the LLM planner aren't in this offline table. They come from the orchestrator runs (`evaluate_v2.py`).

## 5. Grading-scale consistency of explanations (fixed before the final v2 run)

The EyePACS labels use the **International Classification of DR** (ICO 2017, Table 1, PDF p.7). In v1,
retrieval for grades 1–3 mostly returned AAO PPP Appendix 5 (PDF pp. 60–65), which is the **ETDRS**
severity scale ("≥ standard photograph 2A", "fields 4 through 7"). That's a different scale.
**880 of 2,240 v1 explanations (39%) quoted ETDRS criteria.** The text was genuine guideline content,
so nothing was invented, but it came from the wrong scale for these labels.

v2 fix:
- The International Classification definition of the predicted grade (and of the alternate grade) is passed to the LLM as a fixed fact, verbatim from ICO Table 1. This is the same "facts in code, synthesis by LLM" principle already used for the referral timeframes.
- The verifier rejects ETDRS-scale terms and asks for a regeneration.
- Retrieved excerpts may still add context, but only if it agrees with the official definition.

Referral timeframes: verified identical to ICO 2017 Table 2a (high resource) and Table 2b (low/intermediate), PDF p.8.
Page numbers in citations are **PDF page numbers** (ICO: PDF p.7 = printed page 2; PDF p.8 = printed page 3). Say so in the report.

## 6. Final v2 orchestrator run (test set, 2,240 images; LLM planner + explanations)

Files: `v2_llm.csv` and `v2_llm.json`. Grading is unchanged by orchestration (κ 0.8719, accuracy 0.7268; same models).
Summary tables: `v2_ablation_table.csv`, `v2_planner_analysis.csv`, `v2_explanation_quality.csv` (from `evaluate_v2.py`).

### 6.1 Triage

| Test policy | Auto-cleared | Missed grade ≥ 2 | Miss rate | Referral recall | Workload | Verified referrals |
|---|---|---|---|---|---|---|
| Midsem (v1 gate, no verification) | 396 | 16 | 4.04% | 98.62% | 82.32% | 0 |
| v2 gate, no verification | 423 | 13 | 3.07% | 98.88% | 81.12% | 0 |
| v2 gate + lesion check | 405 | 12 | 2.96% | 98.97% | 81.92% | 18 |
| v2 gate + lesion + fellow eye (offline, P4) | 396 | 10 | 2.53% | 99.14% | 82.32% | 27 |
| v2 gate + lesion + all tools (fixed rule) | 352 | 10 | 2.84% | 99.14% | 84.29% | 71 |
| **v2 gate + lesion + LLM planner** | **352** | **10** | **2.84%** | **99.14%** | **84.29%** | 71 |

What each check contributed, among the 423 auto-clear candidates (first discordant check per case):

| Check | Overrides (clear → refer) | Truly referable among them |
|---|---|---|
| Lesion check (MA_count > 11) | 18 | 1 |
| Fellow eye (other eye graded ≥ 2) | 9 | 2 |
| TTA (P(non-referable) under TTA < 0.94) | 44 | **0** |

- **Lesion check and fellow eye** caught 3 of the 13 misses at a cost of 27 extra referrals.
- **TTA caught nothing and added 44 referrals.** In all 44 cases TTA still predicted grade 0 or 1. Only P(non-referable) dipped below the gate (range 0.797–0.94). So TTA as specified just re-tests the confidence gate under noise. It is redundant with the gate, not independent evidence. Report this as a negative result. Don't re-tune the TTA rule on test data. A rule like "TTA counts as discordant only if it predicts grade ≥ 2" would need selecting on val, and is future work.

### 6.2 LLM planner behaviour (honest result)

- In 662 planner decisions the LLM **never chose "stop"**. It called every available tool, always fellow eye first. Its tool sequence matched the fixed rule in 100% of cases, and so did its outcomes (verified by the `all_tools` run: identical 352 / 10 / 662).
- Repeated runs gave identical results at temperature 0.1.
- Its stated reasons were sometimes factually wrong. In 23 of 662 decisions it claimed the lesion count was "above threshold", but the planner is only reached after the lesion check has passed.
- Interpretation: with a safety-first goal ("a missed referable case is far more costly"), the LLM's discretion collapses to "gather all evidence". It adds no measurable value over a fixed policy, and its rationales can't be trusted as explanations of the decision. This supports the design choice that **outcomes are decided in code**: the LLM can only request more evidence, and verification can only make a decision more conservative. Every clinical decision (grade, gate, checks, referral) is identical without the LLM. The LLM's role is the explanation.

### 6.3 Explanation quality (verifier rules v2.1)

| | v1 explanations (re-scored with v2.1 rules) | v2 explanations |
|---|---|---|
| Pass all rules | 0.98% | **90.62%** (2,030 / 2,240) |
| Use ETDRS-scale criteria | 880 (39%) | **13 (0.6%)** |
| Generation attempts | n/a | 1: 1,199 · 2: 676 · 3: 365 (mean 1.63) |

The v1 comparison is indicative only: v1 explanations were written under a different prompt, without these rules.
Remaining v2 failures (accepted after 3 attempts): "indicating/suggests" 165, more than 4 sentences 123, ETDRS terms 13, missing citation 7, truncated 6, restates grade 5, lesion count mismatch 2, "backbone detected" 1.
Lesion counts quoted in explanations match the measurements in all but 2 of 2,240.

**Next-steps sentence (deterministic, no LLM).** In the original run it was built from the predicted grade only, so 757 grade 0/1 cases that were referred (REFER_LOW_CONFIDENCE or REFER_VERIFIED) said "referral is not required".
It now follows the final decision: it states the referral and gives the ICO interval for reference.
The saved outputs were rewritten with `agentic/fix_next_steps.py`. No other column changed, and no reported metric uses this field.

## 7. Visualising agency: state-dependent tool use (for the GUI demo and figures)

Each image's tool sequence depends on what the agent observes. It is not random: the same image always
takes the same path, which is reproducible by design. The agent stops at the first discordant check.
Paths taken on the test set (`v2_llm.csv`, `verification` column):

| Path | Images |
|---|---|
| Predicted grade ≥ 2 → REFER_SEVERITY, no verification | 1,131 |
| P(non-referable) < 0.94 → REFER_LOW_CONFIDENCE, no verification | 686 |
| Lesion ✓ → fellow eye ✓ → TTA ✓ → AUTO_CLEAR | 231 |
| Lesion ✓ → TTA ✓ → AUTO_CLEAR (no fellow eye available) | 121 |
| Lesion ✓ → fellow eye ✓ → TTA ✗ → REFER_VERIFIED | 26 |
| Lesion ✗ → REFER_VERIFIED | 18 |
| Lesion ✓ → TTA ✗ → REFER_VERIFIED | 18 |
| Lesion ✓ → fellow eye ✗ → REFER_VERIFIED | 9 |

That is eight distinct paths using 0 to 3 tools. The variation comes from the evidence, through
state-dependent control flow, not from LLM randomness.

**Demo / figure cases.** In all three, the classifier predicted grade 0 confidently, so it would have been auto-cleared, but the true grade is 2 (moderate NPDR):

| Main image | Fellow-eye upload | What the trace shows |
|---|---|---|
| `14811_right` | none needed | Lesion check: MA_count = 37 (threshold 11) → discordant → refer |
| `40260_right` | `40260_left` | Lesion ✓ (5), fellow eye graded moderate NPDR (0.614) → refer |
| `28409_right` | `28409_left` | Lesion ✓ (6), fellow eye graded moderate NPDR (0.506) → refer |

Contrast cases:
- `41055_left` with its fellow eye: all three checks pass → AUTO_CLEAR (true grade 0).
- `4877_left`: predicted grade 2 → referred immediately, no tools (true grade 3).
- Use "Reveal true grade" after each run to confirm.

**Suggested answer to "what is agentic about this work?"**
The system is given a goal: auto-clear as many patients as is safe. It works towards that goal in a
loop. When a case looks clearable, it gathers more evidence with tools (lesion segmentation, fellow eye,
TTA) and acts on what it finds, referring if any evidence is discordant. Every decision is enforced in
code and can only ever become more cautious. An LLM writes a cited explanation for each case, which a
verifier checks and sends back for correction. Letting the LLM choose which checks to run was also
tested and added nothing over a fixed rule, so decisions stay in code by design.
Follow-up ("is the LLM needed?"): not for decisions, which is deliberate. It is needed for the
explanation, and both roles were measured.

## 8. What 5-class grading adds over a binary output (vs LumineticsCore)

Output: `urgency_analysis.csv`; script: `agentic/urgency_analysis.py`.

**Commercial systems, checked against sources (correct the midsem claim that existing systems are only binary):**
- **US LumineticsCore (formerly IDx-DR), FDA:** binary. The outputs are "mtmDR detected: refer to an eye care professional", "mtmDR NOT detected: retest in 12 months", or insufficient quality (digitaldiagnostics.com product page; pivotal trial protocol NCT02963441). The pivotal trial measured vtDR sensitivity separately, but the device itself outputs only mtmDR.
- **EU IDx-DR v2:** three levels: negative, mtmDR and vision-threatening DR (vtDR) (Spanish validation study, PMC8120039).
- **EyeArt:** covered in reference [13] (Teng et al.). It was FDA-cleared (2020) for mtmDR **and** vision-threatening DR. Internally it grades on the International Clinical 5-level scale, but its output collapses to non-referable vs referable (mtmDR, vtDR). [13] says the purpose is "to assign time-sensitivity to the patients with vision-threatening DR when managing referrals". Its vtDR sensitivity of 88.9–100% and specificity of 89.8–97.5% come from the FDA 510(k) summary K200667. Add that as a reference only if you quote these figures.
- **The midsem cited [13] for "existing systems output a binary referable / non-referable decision". That is wrong for EyeArt, so correct it in the final report.**
- vtDR in these systems = severe NPDR, PDR **and/or macular edema**. EyePACS has no DME labels, so this project's "urgent" (grade ≥ 3) excludes DME. State this limitation.

**So the novelty is granularity, not "urgency vs none":** commercial outputs have 2 or 3 levels. The 5-class grade maps onto all five ICO 2017 Table 2a intervals: grade 0 → 1–2 years, 1 → 6–12 months, 2 → 3–6 months, 3 → < 3 months, 4 → < 1 month. That separates PDR (< 1 month) from severe NPDR (< 3 months), and mild NPDR (6–12 months) from no DR (1–2 years), which a 3-level output can't. It also allows ordering the referral queue by predicted urgency.

| Test set | Value |
|---|---|
| Vision-threatening DR (grade ≥ 3): sensitivity / specificity / AUC | 85.5% / 95.6% / 0.972 |
| Among truly referable cases: AUC for urgent (3–4) vs moderate (2) | 0.944 |
| Assigned the exactly correct ICO interval | 72.7% |
| Assigned an interval within one step of correct | 94.4% |
| Assigned a later interval than correct (under-triage) | 17.0% (≥ 2 steps later: 3.5%) |
| True severe NPDR given an urgent interval (< 3 months) | 80.7% |
| True PDR given an urgent interval / exactly < 1 month | 90.8% / 70.2% |
| Referral queue ordered by P(grade ≥ 3): share of true urgent cases in the first half | 90.0% (random ordering: 50%) |

Val gives the same picture: urgent AUC 0.977, 91.5% of urgent cases in the first half of the queue.

**Framing:** binary (mtmDR) performance is in the same range as deployed systems. vtDR-equivalent
performance (85.5% / 95.6%) is in a similar range to EyeArt's published vtDR figures. Those are not
directly comparable: different data, and theirs includes DME. Beyond that, the 5-class grade gives the
full ICO interval and lets the referral queue be **prioritised**: specialists seeing patients in order of predicted urgency would see 90% of severe and
proliferative cases in the first half of their list, and 91% of PDR patients are flagged as urgent.
QWK (0.872) is the standard summary metric. Sensitivity, specificity and AUC for vision-threatening DR are the clinically
interpretable ones. Report under-triage honestly: 17% of cases get a later interval than their true grade,
mostly by one step.

## 9. Statistics: confidence intervals, significance, screening projection

Every table, with 95% patient-level bootstrap CIs: `STATS_SUMMARY.md` and `stats/*.csv`, made by
`agentic/thesis_statistics.py` (35 tables: confusion matrices, per-class metrics, calibration, binary
endpoints for every model, triage policies, decision × grade tables, verification, lesion, explanations,
saliency, bilateral, urgency, dataset, weight sweep).

**Report honestly: the safety gains point the right way but are not statistically significant at n = 2,240.**
- Misses avoided, v1 gate → final system: **6 [95% CI 0, 13]**, exact McNemar p = 0.11 (8 caught only by the final system, 2 only by v1).
- v2 gate → final system: 3 [0, 7], p = 0.25.
- With 10–16 events per policy a test is underpowered. State the direction, the CI and the trade-off curve (section 4), and name "larger or multi-site evaluation" as needed to confirm.
- Ensemble vs ResNet-50 on exact grade *is* significant on test: McNemar p = 0.0008 (138 vs 87 discordant pairs). Ensemble vs ViT: p < 1e-6.

**Auto-clear rate: the test set is enriched, so project to screening prevalence.**
Test has 51.8% referable. The full 88k EyePACS pool has 19.3% (`L_dataset`). Re-weighting each policy's
per-grade clear rates to the pool's grade mix (`E_screening_projection`):

| Policy | Auto-clear on test | Projected auto-clear at screening prevalence | Missed referable per 1,000 screened |
|---|---|---|---|
| v1 gate (midsem) | 17.7% | 39.2% | 4.1 |
| v2 gate | 18.9% | 46.0% | 3.4 |
| Final system | 15.7% | 39.4% | **2.6** |

The final system keeps about the same specialist workload as the midsem policy at realistic prevalence,
with **37% fewer missed referable patients per 1,000 screened** (4.1 → 2.6). This assumes per-grade
behaviour transfers to the unselected pool, which is not quality-filtered.

**Other findings worth a sentence:**
- **Attention agreement is not a reliability signal.** The two backbones point to the same quadrant only 33.7% of the time, and accuracy is the same whether they agree (71.9%) or not (73.1%). This supports the explanation rule "report attention, never interpret it".
- **Bilateral:** 84.4% of test patients have the same grade in both eyes, and in 7.7% one eye is referable and the other isn't. This justifies using the fellow eye as evidence, not averaging the two eyes.
- **Calibration:** the binary P(non-referable) that the gate uses is well calibrated (test ECE 0.021). That is the property that matters for the gate.
- **Explanations:** pass rate is lowest for PDR predictions (80.2%, mean 1.95 attempts) and highest for REFER_VERIFIED (95.8%).
