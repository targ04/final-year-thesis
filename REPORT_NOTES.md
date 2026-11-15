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

### 6.1 Triage

| Test policy | Auto-cleared | Missed grade ≥ 2 | Miss rate | Referral recall | Workload | Verified referrals |
|---|---|---|---|---|---|---|
| Midsem (v1 gate, no verification) | 396 | 16 | 4.04% | 98.62% | 82.32% | 0 |
| v2 gate only | 423 | 13 | 3.07% | 98.88% | 81.12% | 0 |
| v2 gate + lesion check | 405 | 12 | 2.96% | 98.97% | 81.92% | 18 |
| v2 gate + lesion + fellow eye (offline, P4) | 396 | 10 | 2.53% | 99.14% | 82.32% | 27 |
| **v2 gate + lesion + LLM planner (fellow eye, TTA)** | **352** | **10** | **2.84%** | **99.14%** | **84.29%** | 71 |

What each check contributed, among the 423 auto-clear candidates (first discordant check per case):

| Check | Overrides (clear → refer) | Truly referable among them |
|---|---|---|
| Lesion check (MA_count > 11) | 18 | 1 |
| Fellow eye (other eye graded ≥ 2) | 9 | 2 |
| TTA (P(non-referable) under TTA < 0.94) | 44 | **0** |

- **Lesion check and fellow eye** caught 3 of the 13 misses at a cost of 27 extra referrals.
- **TTA caught nothing and added 44 referrals.** In all 44 cases TTA still predicted grade 0 or 1. Only P(non-referable) dipped below the gate (range 0.797–0.94). So TTA as specified just re-tests the confidence gate under noise. It is redundant with the gate, not independent evidence. Report this as a negative result. Don't re-tune the TTA rule on test data. A rule like "TTA counts as discordant only if it predicts grade ≥ 2" would need selecting on val, and is future work.

### 6.2 LLM planner behaviour (honest result)

- In 662 planner decisions the LLM **never chose "stop"**. It called every available tool, always fellow eye first. Its choices were therefore identical to the fixed rule "run all tools" in 100% of cases, and so were its triage outcomes.
- Repeated runs gave identical results at temperature 0.1: the `llm` triage-only run, the full run and a duplicate run produced the same 352 / 10 / 662.
- Its stated reasons were sometimes factually wrong. In 23 of 662 decisions it claimed the lesion count was "above threshold", but the planner is only reached after the lesion check has passed.
- Interpretation: with a safety-first goal ("a missed referable case is far more costly"), the LLM's discretion collapses to "gather all evidence". It adds no measurable value over a fixed policy, and its rationales can't be trusted as explanations of the decision. This supports the design choice that **outcomes are decided in code**: the LLM can only request more evidence, and verification can only make a decision more conservative. The agentic contribution is the goal-directed verification loop, meaning tools, a code-enforced one-way override and a verify-and-feedback cycle. It is not LLM judgement.

### 6.3 Explanation quality (verifier rules v2.1)

| | v1 explanations (re-scored with v2.1 rules) | v2 explanations |
|---|---|---|
| Pass all rules | 0.98% | **90.62%** (2,030 / 2,240) |
| Use ETDRS-scale criteria | 880 (39%) | **13 (0.6%)** |
| Generation attempts | n/a | 1: 1,199 · 2: 676 · 3: 365 (mean 1.63) |

The v1 comparison is indicative only: v1 explanations were written under a different prompt, without these rules.
Remaining v2 failures (accepted after 3 attempts): "indicating/suggests" 165, more than 4 sentences 123, ETDRS terms 13, missing citation 7, truncated 6, restates grade 5, lesion count mismatch 2, "backbone detected" 1.
Lesion counts quoted in explanations match the measurements in all but 2 of 2,240.
