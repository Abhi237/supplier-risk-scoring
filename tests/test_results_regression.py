"""Metric regression on the committed real-data artifacts (skipped when artifacts are absent, e.g. fresh clone
without `make all`). Guards against silent changes to the published numbers."""
import json
from pathlib import Path

import pandas as pd
import pytest

ART = Path(__file__).resolve().parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "train_summary.json").exists(), reason="no pipeline artifacts")


def load(n):
    return json.loads((ART / n).read_text())


def test_data_integrity_counts():
    s = load("sql_summary.json")
    assert s["dq"]["orders_rows"] == 99441
    assert s["dq"]["order_items_rows"] == 112650
    assert s["dq"]["sellers_rows"] == 3095
    assert s["n_sellers_active"] == 3095


def test_headline_rates():
    s = load("sql_summary.json")
    assert s["order_late_rate"] == pytest.approx(0.0677, abs=5e-4)
    assert s["late_rate_if_handoff_late"] > 3 * s["late_rate_if_handoff_on_time"]


def test_oot_metrics_stable():
    ev = pd.read_csv(ART / "oot_metrics_main.csv").set_index(["target", "score"])
    ho = ev.loc["Seller hand-off late"]
    assert ho.loc["Shrunk hand-off late rate (EB) only", "item_auc"] == pytest.approx(0.704, abs=0.005)
    assert ho.loc["Composite risk score (shrunk)", "item_auc"] > ho.loc["Baseline: volume only (smaller = riskier)",
                                                                       "item_auc"]
    assert ho.loc["Shrunk hand-off late rate (EB) only", "item_auc"] > ho.loc["Baseline: raw hand-off late rate",
                                                                             "item_auc"]


def test_shrinkage_reduces_small_seller_error():
    c = pd.read_csv(ART / "oot_shrinkage_calibration.csv").set_index("vol_band")
    assert c.loc["<5", "mae_shrunk"] < c.loc["<5", "mae_raw"]


def test_scores_cover_all_sellers():
    t = load("train_summary.json")
    assert t["n_sellers_scored"] == 3095
    assert sum(r["n_sellers"] for r in pd.read_csv(ART / "risk_tier_summary.csv").to_dict("records")) == 3095
