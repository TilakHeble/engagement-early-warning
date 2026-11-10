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

## 2026-11-10: stage 3, landmark model of disengagement onset (`notebooks/03_landmark_onset.ipynb`)

- **Reframed target.** The event is now *disengagement onset*: a withdrawing student's last VLE activity falls
  within the next 4 weeks. Students already permanently silent count as negatives (in practice they look like
  students on a break, and the goal is warning before silence). `last_active_day` is used only for labels.
- **Context check.** A 2026 OULAD survival benchmark (arXiv:2604.08870) used the unregistration date as the event,
  split by student (shared modules across train and test), reported no capacity or lead-time metrics, and left
  landmarking to future work. This stage addresses all four.
- **Model.** One pooled "landmark super-model" over weeks 1–24 (616,946 student-weeks; 2.9% events), with spline
  week terms plus feature-by-week interactions (logistic), and gradient boosting with week as a feature.
  5-fold CV grouped by presentation.
- **Results** (weeks 4–12, 10% capacity):
  - Gradient boosting catches 23–28% of upcoming disengagements (random would catch 10%; AUC 0.69–0.72), a median
    of 1.3–2.3 weeks before the student goes silent.
  - The baseline falls to about random from week 8: low engagement marks students already gone, not those about to go.
  - Against the stage 2 model, precision rises only +1.0–1.1 pp (paired intervals exclude 0 at weeks 8 and 12 only).
    Ranking gains are small. The main gains are calibration (predicted vs observed within about 1 point per
    decile; the stage 2 scores are 5–38% where onset rates are 1–9%) and fewer contacts wasted on already-silent
    students (6–8% vs 9–13%).
  - Trajectory and recency retested in the nonlinear model: no gain (within ±0.3 pp, intervals include 0).
    The stage 2 negative result stands.
- **Chosen stage 3 model:** landmark gradient boosting, with landmark logistic as the interpretable comparator.
