"""Re-run the whole pipeline from the raw OULAD tables to every results file.

    python -m eew.reproduce            # about 15-20 minutes on a laptop
    python -m eew.reproduce --quick    # skip the slow transfer experiments (stage 6)

Steps: processed tables -> stage 1 baselines -> stage 2 features -> stage 3 landmark data and
held-out predictions -> stages 3-6 experiments -> dashboard snapshot. Outputs go to
data/processed/, results/ and dashboards/snapshot/. The notebooks read the same functions, so
`make notebooks` re-executes them on top of these results.
"""

from __future__ import annotations

import argparse
import time
import warnings


def main(quick: bool = False) -> None:
    from eew import data, experiments as X
    from eew.dashboard_data import build_snapshot

    warnings.filterwarnings("ignore")
    steps = [
        ("Processed tables", lambda: data.build_all()),
        ("Stage 1: baselines", X.run_stage1),
        ("Stage 2: features vs baseline", X.run_stage2),
        ("Stage 3: landmark data (rebuild)", lambda: X.build_landmark_data(cache=False)),
        ("Stage 3: held-out predictions (rebuild)", lambda: X.landmark_predictions(cache=False)),
        ("Stage 3: onset model comparison", X.run_stage3),
        ("Stage 4: outreach simulation", X.run_stage4),
        ("Stage 5: fairness audit", X.run_stage5),
    ]
    if not quick:
        steps += [("Stage 6: transfer", X.run_stage6), ("Stage 6: module calibration", X.run_stage6_calibration)]
    steps.append(("Stage 7: dashboard snapshot", build_snapshot))
    start = time.time()
    for name, fn in steps:
        t = time.time()
        fn()
        print(f"[{time.time() - start:6.0f}s] {name} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip stage 6 transfer experiments")
    main(ap.parse_args().quick)
