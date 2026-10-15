"""Streamlit AppTest: every page renders without exceptions from the committed app/data/ artifacts."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
DATA = APP.parent / "data" / "summary.json"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="app/data not exported")

PAGES = [("views/overview.py", "Empirical-Bayes shrinkage to score on-time delivery risk"),
         ("views/data_quality.py", "Data & quality"),
         ("views/delivery.py", "Delivery performance"), ("views/scorecards.py", "Vendor scorecards"),
         ("views/validation.py", "Risk model & validation"), ("views/archetypes.py", "Indicative vendor archetypes"),
         ("views/exposure.py", "Exposure & scenario simulator"), ("views/methodology.py", "Methodology & limitations")]


def _open(page):
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    if page != "views/overview.py":
        at.switch_page(page).run()
    return at


@pytest.mark.parametrize("page,title", PAGES)
def test_page_renders(page, title):
    at = _open(page)
    assert not at.exception, [e.value for e in at.exception]
    assert title in at.title[0].value


@pytest.mark.parametrize("page,_t", PAGES)
def test_page_states_scope_and_no_emoji(page, _t):
    at = _open(page)
    text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [t.value for t in at.title])
    assert "hand-off" in text.lower() or "handoff" in text.lower() or page in ("views/exposure.py",)
    assert all(ord(ch) < 0x2190 or ch in "\u2264\u2265\u00b7\u00d7" for ch in text), "unexpected non-text symbol in page text"


def test_scorecard_search_filters():
    at = _open("views/scorecards.py")
    at.text_input[0].input("zzzz-no-such-vendor").run()
    assert not at.exception
    assert any("No vendor matches" in i.value for i in at.info)


def test_scorecard_search_finds_vendor():
    at = _open("views/scorecards.py")
    at.text_input[0].input("cipla").run()
    assert not at.exception
    assert any("CIPLA" in x.value for x in at.subheader)


def test_validation_target_toggle():
    at = _open("views/validation.py")
    at.radio[0].set_value(at.radio[0].options[1]).run()
    assert not at.exception


def test_simulator_reacts_to_assumptions():
    at = _open("views/exposure.py")
    base = at.metric[3].value
    at.slider[0].set_value(0.20).run()
    assert not at.exception
    assert at.metric[3].value != base


def test_reallocation_zero_volume_moves_nothing():
    at = _open("views/exposure.py")
    frac = [s for s in at.slider if "volume moved" in s.label][0]
    frac.set_value(0.0).run()
    assert not at.exception
    assert any(m.label.startswith("Late avoided") and m.value.startswith("$0") for m in at.metric)
