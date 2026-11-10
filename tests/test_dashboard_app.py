"""Stage 7: the dashboard runs without errors on the committed snapshot."""

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "dashboards" / "app.py"
SNAPSHOT = APP.parent / "snapshot" / "students.parquet"


@pytest.mark.skipif(not SNAPSHOT.exists(), reason="dashboard snapshot not built")
def test_app_runs_across_settings():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    at.sidebar.toggle[0].set_value(True).run()  # reveal outcomes
    assert not at.exception
    for module in ["AAA", "CCC", "GGG"]:
        at.sidebar.selectbox[0].set_value(module).run()
        assert not at.exception
    at.sidebar.slider[0].set_value(24).run()
    at.sidebar.select_slider[1].set_value(0).run()  # no cool-down
    assert not at.exception
