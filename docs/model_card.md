# Model card: Engagement Early-Warning (disengagement onset)

**Status:** research prototype on open data. Not a production system and not affiliated with any company or product.
**Live demo:** https://engagement-early-warning.streamlit.app/

## What it predicts
Each week of a module, for each student still enrolled: the probability that the student goes silent for good on the
VLE within the next 4 weeks and later withdraws ("disengagement onset"). Official withdrawal dates lag real
disengagement (median 9 days; 14+ days for a third of students), so the paperwork date is not the target.

## Intended use
- Ranking students each week so a support team with limited capacity contacts those most likely to disengage soon.
- **Not** for automated decisions about students, assessment, progression or funding.
- **Not** a diagnosis of a student's circumstances; warning signs are prompts for a conversation.

## Data
Open University Learning Analytics Dataset (OULAD; Kuzilek, Hlosta & Zdrahal, 2017; CC BY 4.0): 32,593 enrolments,
7 modules, 22 presentations (2013–2014), 10.6M daily click records. Withdrawals before the module starts (26%) are
out of scope: no in-course behaviour exists to predict them.

## Model
Pooled landmark gradient boosting (scikit-learn `HistGradientBoostingClassifier`) over weeks 1–24 (617k student-weeks),
with week as a feature. 23 features in 7 groups: level, regularity, recency, trajectory, cohort-relative engagement,
assessment behaviour (missed, late, released scores), and study context. **No demographic features.**

## Evaluation (held-out module presentations; 10% weekly contact capacity)
| Measure | Result |
| --- | --- |
| Upcoming disengagements caught, weeks 4–12 | 23–28% (random list: 10%) |
| Median warning before the student goes silent | 1.3–2.3 weeks |
| AUC | 0.69–0.72 |
| Calibration, pooled | within ~1 pt per decile |
| Term coverage at 5%/week, 4-week cool-down | 51% of withdrawers reached before silence (random 40%, baseline 29%) |
| Module never seen in training | 22.9% caught (vs 23.9%) |

## Known limitations and risks
- **Under-predicts risk for students with a declared disability** (calibration 0.75 vs 1.06). Students aged 35+ and those
  with a degree are missed 6–8 points more often. A recalibration-by-disability option fixes calibration but uses a
  protected characteristic (Equality Act 2010): an institutional decision, not a default.
- **Probabilities are not calibrated per module** (0.7–2.4× off for a new module). Use the ranking from day one; treat
  probabilities as provisional until one completed run allows per-module recalibration (with shrinkage).
- Evaluation measures who is *reached*, not who is *retained*; impact needs a trial.
- One institution's data; another VLE and other data definitions may behave differently.
- "Drop from own baseline" (trajectory) features add nothing measurable: a negative result, kept for transparency.

## Ethics
Students should be told what data is used and why; outreach should be supportive, not punitive; staff should be able to
override the list; group-level performance should be monitored each term.
