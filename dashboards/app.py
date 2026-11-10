"""Engagement Early-Warning: tutor dashboard (research prototype).

Run locally:   streamlit run dashboards/app.py
Data:          dashboards/snapshot/ (held-out predictions for the 2014 OULAD presentations)
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

SNAP = Path(__file__).parent / "snapshot"
KEY = ["code_module", "code_presentation", "id_student"]
TEAL, ORANGE, GREY, RED = "#2a9d8f", "#DD8452", "#8d99ae", "#e76f51"

st.set_page_config(page_title="Engagement Early-Warning", page_icon="📈", layout="wide")


@st.cache_data
def load():
    students = pd.read_parquet(SNAP / "students.parquet")
    tables = {n: pd.read_csv(SNAP / f"{n}.csv") for n in [
        "stage4_curves", "stage4_cooldowns", "stage5_group_metrics", "stage6_unseen_module",
        "stage6_module_calibration", "stage3_metrics"]}
    return students, tables


def weekly_lists(pres: pd.DataFrame, capacity: float, cooldown: int, upto: int) -> dict[int, set]:
    """Replay weekly outreach for one presentation: who is contacted each week (top risk, with cool-down)."""
    last: dict[int, int] = {}
    lists = {}
    for week in range(1, upto + 1):
        g = pres[pres.week == week]
        if g.empty:
            continue
        n = int(np.ceil(capacity * len(g)))
        eligible = g[[week - last.get(s, -10_000) > cooldown for s in g.id_student]]
        chosen = eligible.nlargest(n, "risk").id_student
        for s in chosen:
            last[s] = week
        lists[week] = set(chosen)
    return lists


def outcome_label(r) -> str:
    if r.onset:
        return f"Went silent within {max(r.weeks_to_onset, 0):.0f}–{max(r.weeks_to_onset, 0) + 1:.0f} weeks, then withdrew"
    if r.already_silent:
        return "Had already gone silent (withdrew)"
    if r.will_withdraw:
        return "Withdrew later in the module"
    return f"Stayed to the end ({r.final_result})"


students, T = load()

# ----------------------------------------------------------------------------------- sidebar
st.sidebar.title("Weekly outreach")
modules = sorted(students.code_module.unique())
module = st.sidebar.selectbox("Module", modules, index=modules.index("DDD") if "DDD" in modules else 0)
presentations = sorted(students[students.code_module == module].code_presentation.unique())
presentation = st.sidebar.selectbox("Presentation", presentations, index=len(presentations) - 1)
week = st.sidebar.slider("Week of the module", 1, 24, 6)
capacity = st.sidebar.select_slider("Contact capacity (share of cohort per week)",
                                    options=[0.025, 0.05, 0.10], value=0.05, format_func=lambda x: f"{x:.1%}")
cooldown = st.sidebar.select_slider("Cool-down before re-contacting (weeks)", options=[0, 2, 4, 8], value=4)
reveal = st.sidebar.toggle("Reveal what actually happened (demo only)", value=False,
                           help="Outcomes are only known after the fact. Shown here to let you judge the list.")
st.sidebar.caption("Research prototype on the open OULAD dataset (CC BY 4.0). Independent project, "
                   "not affiliated with any company or product.")

pres = students[(students.code_module == module) & (students.code_presentation == presentation)]
this_week = pres[pres.week == week]

st.title("Engagement Early-Warning")
st.markdown("Which students are **about to stop engaging**? Each week the model ranks students still enrolled by their "
            "risk of going silent for good in the next 4 weeks, so a support team can reach them first.")

tab_list, tab_plan, tab_trust, tab_about = st.tabs(
    ["This week's list", "Planning capacity", "How far can I trust it?", "About & limitations"])

# ----------------------------------------------------------------------------------- tab 1
with tab_list:
    lists = weekly_lists(pres, capacity, cooldown, week)
    flagged_ids = lists.get(week, set())
    flagged = this_week[this_week.id_student.isin(flagged_ids)].sort_values("risk", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students enrolled this week", f"{len(this_week):,}")
    c2.metric("To contact this week", f"{len(flagged):,}")
    top_n = this_week.nlargest(int(np.ceil(capacity * len(this_week))), "risk").id_student
    c3.metric("High-risk but contacted recently", f"{int((~top_n.isin(flagged_ids)).sum()):,}",
              help="Students in this week's top risk band who are skipped because of the cool-down.")
    c4.metric("Average 4-week risk on the list", f"{flagged.risk.mean():.1%}" if len(flagged) else "–",
              help="Probabilities are provisional: they are calibrated overall, but can be off by up to 2× for an "
                   "individual module until it has one completed run (see 'How far can I trust it?').")

    show = flagged.assign(
        rank=np.arange(1, len(flagged) + 1), id_student=flagged.id_student.astype(str),
        risk_pct=(flagged.risk * 100).round(1),
        signs=flagged.warning_signs.str.split(" \\| "),
    )
    cols = {"rank": "#", "id_student": "Student", "risk_pct": "4-week risk (%)", "signs": "Warning signs"}
    if reveal:
        show["outcome"] = show.apply(outcome_label, axis=1)
        cols["outcome"] = "What happened"
    st.dataframe(
        show[list(cols)].rename(columns=cols), hide_index=True, width="stretch",
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Student": st.column_config.TextColumn(width="small"),
            "4-week risk (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                               max_value=max(30.0, float(show.risk_pct.max() if len(show) else 30))),
            "Warning signs": st.column_config.ListColumn(width="large"),
        },
        height=min(560, 38 * (len(show) + 1) + 4),
    )
    st.caption("Warning signs are simple observable signals to open a conversation (missed or late assessments, "
               "silence on the VLE, low or falling activity compared with the cohort or the student's own earlier weeks). "
               "They are not the model's internal reasoning.")

    if reveal and len(flagged):
        hits = flagged.onset.sum()
        events = this_week.onset.sum()
        st.info(f"**This week:** {hits} of the {len(flagged)} students on the list went silent for good within 4 weeks "
                f"({hits / len(flagged):.0%}). Across the whole cohort, {events} students did "
                f"({events / len(this_week):.1%}): a random list of the same size would expect about "
                f"{len(flagged) * events / max(len(this_week), 1):.1f} hits.")

    # Term so far: how many withdrawing students have been reached before going silent?
    reached = set().union(*lists.values()) if lists else set()
    st.divider()
    st.subheader("Contacts so far this term")
    weekly = pd.DataFrame({"week": list(lists), "contacted": [len(v) for v in lists.values()]})
    chart = alt.Chart(weekly).mark_bar(color=TEAL).encode(
        x=alt.X("week:O", title="Week"), y=alt.Y("contacted:Q", title="Students contacted"))
    st.altair_chart(chart.properties(height=180), width="stretch")
    st.caption(f"{len(reached):,} different students contacted in weeks 1–{week} "
               f"({len(reached) / max(pres.id_student.nunique(), 1):.0%} of the cohort).")

# ----------------------------------------------------------------------------------- tab 2
with tab_plan:
    st.subheader("How much capacity do you need?")
    st.markdown("A whole term (weeks 1–24) of weekly outreach, replayed on held-out scores across all 22 OULAD "
                "presentations. **Coverage** = share of withdrawing students contacted *before they went silent*.")
    cur = T["stage4_curves"].copy()
    cur["reached"] = cur.reached_before_silence * 100
    sel = cur[(cur.capacity.round(3) == round(capacity, 3))]
    line = alt.Chart(cur).mark_line(point=True).encode(
        x=alt.X("contacts_per_100_students:Q", title="Workload: contacts per 100 students per term"),
        y=alt.Y("reached:Q", title="% of withdrawers reached before silence", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("score:N", title=None, scale=alt.Scale(
            domain=["Stage 3 model", "Random", "Baseline: cumulative clicks"], range=[TEAL, GREY, ORANGE])),
        tooltip=["score", alt.Tooltip("capacity:Q", format=".1%"), alt.Tooltip("reached:Q", format=".0f")],
    )
    rule = alt.Chart(sel).mark_rule(strokeDash=[4, 4], color="black").encode(x="contacts_per_100_students:Q")
    st.altair_chart((line + rule).properties(height=360), width="stretch")
    m = sel.set_index("score")
    a, b, c = st.columns(3)
    a.metric(f"Model at {capacity:.1%}/week", f"{m.loc['Stage 3 model', 'reached_before_silence']:.0%}",
             f"{(m.loc['Stage 3 model', 'reached_before_silence'] - m.loc['Random', 'reached_before_silence']) * 100:+.0f} pts vs random")
    b.metric("Random list", f"{m.loc['Random', 'reached_before_silence']:.0%}")
    c.metric("Engagement-level baseline", f"{m.loc['Baseline: cumulative clicks', 'reached_before_silence']:.0%}")
    st.markdown("**Cool-down** (model at 5%/week): longer cool-downs reach more students, but a smaller share of "
                "contacts goes to students who withdraw.")
    cd = T["stage4_cooldowns"][["cooldown_weeks", "reached_before_silence", "share_contacted", "contacted_who_withdraw"]]
    st.dataframe(cd.rename(columns={"cooldown_weeks": "Cool-down (weeks)", "reached_before_silence": "Withdrawers reached in time",
                                    "share_contacted": "All students contacted", "contacted_who_withdraw": "Contacts who withdraw"})
                 .style.format({c: "{:.0%}" for c in ["Withdrawers reached in time", "All students contacted", "Contacts who withdraw"]}),
                 hide_index=True)
    st.caption("Assumes contact does not change behaviour: this shows who would be reached, not who would be retained.")

# ----------------------------------------------------------------------------------- tab 3
with tab_trust:
    st.subheader("Performance on held-out cohorts")
    s3 = T["stage3_metrics"]
    s3 = s3[s3.score.isin(["Stage 3 landmark: gradient boosting", "Baseline: cumulative clicks"])].copy()
    s3["model"] = s3.score.map({"Stage 3 landmark: gradient boosting": "Early-warning model",
                                "Baseline: cumulative clicks": "Engagement-level baseline"})
    s3["caught"] = s3.recall * 100
    base = alt.Chart(s3).encode(x=alt.X("week:O", title="Week of the module", axis=alt.Axis(labelAngle=0)))
    ch = base.mark_line(point=True).encode(
        y=alt.Y("caught:Q", title="% caught", scale=alt.Scale(domain=[0, 35])),
        color=alt.Color("model:N", title=None, scale=alt.Scale(
            domain=["Early-warning model", "Engagement-level baseline"], range=[TEAL, ORANGE])),
        tooltip=["model", "week", alt.Tooltip("caught:Q", format=".1f")])
    rnd = alt.Chart(pd.DataFrame({"y": [10]})).mark_rule(strokeDash=[4, 4], color=GREY).encode(y="y:Q")
    st.markdown("**Upcoming disengagements caught with 10% weekly capacity** (dashed line: a random list)")
    st.altair_chart((ch + rnd).properties(height=280), width="stretch")
    st.caption("A random 10% list would catch 10%. Flags come a median of 1.3–2.3 weeks before the student goes silent.")

    left, right = st.columns(2)
    with left:
        st.subheader("Fairness")
        gm = T["stage5_group_metrics"]
        gm = gm[gm.score == "Stage 3 model"].copy()
        gm["Attribute"] = gm.attribute.map({"disability": "Disability", "imd_quintile": "Deprivation (IMD)",
                                            "age_group": "Age", "gender": "Gender", "education": "Prior education"})
        gm["Calibration"] = [f"{c:.2f} [{lo:.2f}–{hi:.2f}]" for c, lo, hi in
                             zip(gm.calibration_ratio, gm.calibration_ratio_lo, gm.calibration_ratio_hi)]
        st.dataframe(gm[["Attribute", "group", "miss_rate", "Calibration"]]
                     .rename(columns={"group": "Group", "miss_rate": "Miss rate"})
                     .style.format({"Miss rate": "{:.0%}"}), hide_index=True, height=420, width="stretch")
        st.caption("Miss rate: upcoming disengagements not flagged. Calibration: predicted ÷ actual risk "
                   "(1 = accurate), with 95% interval.")
        st.warning("Risk is **under-predicted by about a quarter for students with a declared disability**. "
                   "Older students and students with a degree are missed 6–8 points more often.")
    with right:
        st.subheader("New modules")
        um = T["stage6_unseen_module"]
        um = um[um.module == "All"].set_index("setting")
        st.metric("Disengagements caught on a module the model never saw",
                  f"{um.loc['Module never seen', 'recall']:.1%}",
                  f"{(um.loc['Module never seen', 'recall'] - um.loc['Stage 3 validation (module seen)', 'recall']) * 100:+.1f} pts vs module seen")
        mc = T["stage6_module_calibration"].pivot(index="code_module", columns="setting", values="ratio")
        mc = mc[["Cold start (module never seen)", "One-run recalibration (shrunk, k=100)"]]
        mc.columns = ["New module", "After one completed run"]
        mc.index.name = "Module"
        st.markdown("**Predicted ÷ actual risk per module** (1 = accurate)")
        st.dataframe(mc.round(2), width="stretch")
        st.caption("Ranking transfers to new modules; probabilities do not (off by up to 2.4× per module). "
                   "One completed local run, with shrinkage, brings the worst module to within 1.5×.")

# ----------------------------------------------------------------------------------- tab 4
with tab_about:
    st.markdown("""
**What this is.** An independent research prototype built on the open
[Open University Learning Analytics Dataset](https://research.stem.open.ac.uk/ouanalyse/dataset/) (CC BY 4.0;
Kuzilek, Hlosta & Zdrahal, 2017). It is not affiliated with any company or product.

**What the model predicts.** Each week, for each student still enrolled: the probability that they go silent for good
on the VLE within the next 4 weeks and later withdraw ("disengagement onset"). Official withdrawal dates lag real
disengagement (median 9 days), so predicting the paperwork date would reward flagging students who have already gone.

**How it was built and tested.** A pooled landmark gradient-boosting model on 23 behavioural and assessment
features (no demographics), validated with whole module presentations held out. The full analysis, including every
negative result, is in the repository notebooks and build log.

**Limitations.**
- Shows who would be *reached* in time, not who would be *retained*: that needs a trial.
- Probabilities are provisional per module until it has one completed run.
- Risk is under-predicted for students with a declared disability; consider this when reading individual scores.
- One university's data (seven modules); a new institution's VLE and data definitions will differ.
- Warning signs are transparent rules for conversation, not the model's reasoning.

**Snapshot.** The 2014 presentations, weeks 1–24, with held-out (out-of-fold) scores.
""")
