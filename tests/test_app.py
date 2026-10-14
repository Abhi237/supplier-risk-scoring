"""Streamlit AppTest: every page renders without exceptions from the committed app/data/ artifacts."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
DATA = APP.parent / "data" / "summary.json"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="app/data not exported")

PAGES = [("views/overview.py", "greatest delivery-risk"), ("views/data_quality.py", "Data & quality"),
         ("views/delivery.py", "Delivery performance"), ("views/scorecards.py", "Seller scorecards"),
         ("views/validation.py", "Risk model & validation"), ("views/archetypes.py", "Seller archetypes"),
         ("views/exposure.py", "Exposure & scenario simulator"), ("views/methodology.py", "Methodology")]


@pytest.mark.parametrize("page,title", PAGES)
def test_page_renders(page, title):
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    if page != "views/overview.py":
        at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert title in at.title[0].value


def test_scorecard_search_filters():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.switch_page("views/scorecards.py").run()
    at.text_input[0].input("zzzz-no-such-seller").run()
    assert not at.exception
    assert any("No seller matches" in i.value for i in at.info)


def test_simulator_reacts_to_assumptions():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.switch_page("views/exposure.py").run()
    base = at.metric[3].value
    at.slider[0].set_value(0.20).run()   # much higher assumed late-delivery cost
    assert not at.exception
    assert at.metric[3].value != base
