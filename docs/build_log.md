# Build log

## 2026-11-09: project set up

- Scope: predict which students need support before they disengage, on the open OULAD dataset.
- Unit of analysis: one enrolment (student × module presentation).
- Primary outcome: withdrawal, with the withdrawal day taken from `date_unregistration`.
  Secondary outcome: fail or withdraw.
- Prediction points: weeks 2, 4, 6, 8, 10 and 12. A model at week k only sees data from days before 7k.
- Headline metric: share of eventual withdrawals flagged, at a contact capacity of 10% of the cohort per
  week, plus median weeks of warning.
- Done: data loader with schema checks, enrolment table, daily click aggregation, and unit tests.

## 2026-11-09: data loaded and checked

- Source: the seven OULAD tables in Parquet form from the
  [data-edu/oulad](https://github.com/data-edu/oulad) mirror (MIT-licensed package, CC BY 4.0 data),
  because the official download was unreachable from the build environment.
- Checked against the official figures: every table's row count matches (32,593 enrolments;
  10,655,280 VLE rows; 173,912 assessment submissions), and so does the outcome split
  (Pass 12,361; Withdrawn 10,156; Fail 7,052; Distinction 3,024).
- Processed: 1,808,119 enrolment-days with at least one click.
- **Finding:** 2,676 of the 10,156 withdrawals (26%) are dated *before* day 0, and 93 have no date.
  Behaviour inside the course cannot predict these.
- **Decision:** the early-warning population is students still enrolled at day 0. Withdrawals before the
  start, and those with no date, are reported separately rather than mixed into model metrics. Mixing
  them in would quietly inflate or deflate the results.

## 2026-11-09: stage 1, engagement-band baseline (`notebooks/01_eda_and_baseline.ipynb`)

- Population: 29,824 enrolments; 24.8% withdraw; the median withdrawal is in week 9.
- Baselines (only clicks before the week-k cutoff): a 7-day engagement band relative to the cohort
  (none / low / medium / high / very high) and cumulative clicks as a cohort percentile.
- Results at 10% contact capacity, weeks 2–12 (`results/baseline_metrics.csv`):
  - Will withdraw: recall 12.5–15%, AUC 0.59–0.61. Random flagging would give 10%.
  - Will fail or withdraw: AUC 0.66–0.73.
- **Finding:** in clicks, students who will fail look like students who will withdraw, so a level-of-engagement
  score can't tell them apart. It finds "struggling", not specifically "about to leave".
- **Checked:** the low AUC is not a bug. Click totals match an independent merge exactly, and cohort-relative
  percentiles beat raw counts (AUC 0.60 vs 0.56).
- **Decision:** report two targets throughout: `will_withdraw` (retention) and `needs_support` (fail or withdraw).
- Next: stage 2 features (change from the student's own baseline, regularity, recency, assessment
  timeliness), then the stage 3 discrete-time survival model.

## 2026-11-10: evaluation review and fixes

A self-review found three weaknesses in the stage 1 evaluation. All are fixed, and the notebook was re-run:

1. **Warning time was overstated.** `date_unregistration` is an administrative date. The median gap from a
   withdrawn student's last click to official withdrawal is 9 days; it is 14+ days for 34%, and 12% never
   clicked at all. Added `last_activity_day` (evaluation only, never a model input) and report weeks from
   flag to last activity, plus the share of caught students already silent at flag time.
   - Result: baseline flags come about 9 weeks before official withdrawal but only 0–3 weeks before last
     activity. For cumulative clicks, 33–56% of catches (weeks 2–12) were already silent.
2. **No uncertainty.** Added 95% bootstrap intervals, resampling whole module presentations (the
   independent units). Example: withdrawal recall for cumulative clicks at week 4 is 14.9% (14.1–16.0%).
3. **The recall chart was misleading across targets.** At 10% capacity, the maximum recall is roughly
   capacity ÷ base rate, which is about 23% for fail-or-withdraw. The headline chart now shows precision
   against the base rate, and the results table includes `max_recall`.

Also: random tie-breaking checked (withdrawal recall varies only 13.7–14.8% across 20 seeds), the chart
arrow overlap fixed, and Windows run instructions added to the README.

**Implication for later stages:** a useful flag must come *before* a student goes silent. Stage 3 models
are judged on warning time before last activity, not just on precision.

## 2026-11-10: stage 2, research-led features (`notebooks/02_features.ipynb`)

- Features (`src/eew/features.py`), 23 in 7 groups: level, regularity, recency, trajectory (vs the student's
  own earlier activity), cohort-relative, assessment behaviour, and study context. Demographics are never used.
- Leakage controls: only clicks before the cutoff; an assessment counts as due/submitted only before the cutoff;
  scores are used only for submissions made 14+ days before the cutoff (OULAD has no release date). A unit
  test injects future clicks and submissions and checks that no feature changes.
- Model: logistic regression per week, with 5-fold CV **grouped by module presentation**. Gradient boosting
  gave the same AUC, so the simpler model was kept.
- Added `evaluate.paired_difference`: a paired bootstrap over presentations for model-minus-baseline differences.
- **Results (will withdraw, 10% capacity):** precision +6.2 to +10.6 pp over the baseline from week 4 (paired
  intervals exclude 0), e.g. week 8 is 0.33 vs 0.22; AUC 0.66–0.70 vs 0.60. Week 2 shows no reliable gain.
- **Warning before last activity:** 1.6–2.6 weeks (baseline: −1 to −2.4 from week 8). The share of catches
  already silent falls from about 54% to 34–38%.
- **Ablation:** assessment behaviour carries most of the signal (AUC loss 0.04–0.075); cohort-relative
  features come second (0.01–0.018).
- **Negative result:** trajectory (drop from own baseline) and recency add nothing measurable in this linear
  model. Recorded as not supported; to be retested in the stage 3 time-to-event model.
- Coefficients saved (`results/stage2_coefficients.csv`) but not interpreted: correlated features (e.g.
  active_share and its cohort percentile) give unstable, opposite-signed coefficients.
