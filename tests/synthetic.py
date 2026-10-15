"""Test data: a deterministic synthetic SCMS-format raw extract and a handcrafted edge-case table.

Both use the raw column names and text formats of the USAID file (all columns text, 'd-Mon-yy' dates, freight and weight
text values), so the same SQL model runs on them as on the real data.  No real vendor names or values are used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIX = Path(__file__).parent / "fixtures"
COLUMNS = ["ID", "Project Code", "PQ #", "PO / SO #", "ASN/DN #", "Country", "Managed By", "Fulfill Via", "Vendor INCO Term",
           "Shipment Mode", "PQ First Sent to Client Date", "PO Sent to Vendor Date", "Scheduled Delivery Date",
           "Delivered to Client Date", "Delivery Recorded Date", "Product Group", "Sub Classification", "Vendor",
           "Item Description", "Molecule/Test Type", "Brand", "Dosage", "Dosage Form", "Unit of Measure (Per Pack)",
           "Line Item Quantity", "Line Item Value", "Pack Price", "Unit Price", "Manufacturing Site", "First Line Designation",
           "Weight (Kilograms)", "Freight Cost (USD)", "Line Item Insurance (USD)"]
COUNTRIES = ["Zambia", "Nigeria", "Haiti", "Vietnam", "Uganda"]
MODES = ["Air", "Truck", "Ocean"]


def fmt(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day}-{d.strftime('%b-%y')}"


def _row(i, ship, vendor, country, mode, pgroup, sched, deliv, rec, value, first, freight, weight, po="1/1/2010", rdc=False):
    r = dict.fromkeys(COLUMNS, "")
    r.update({"ID": str(i), "Project Code": "100-XX-T01", "PQ #": "Pre-PQ Process", "PO / SO #": f"SCMS-{i}", "ASN/DN #": ship,
              "Country": country, "Managed By": "PMO - US", "Fulfill Via": "From RDC" if rdc else "Direct Drop",
              "Vendor INCO Term": "N/A - From RDC" if rdc else "EXW", "Shipment Mode": mode,
              "PQ First Sent to Client Date": "Pre-PQ Process", "PO Sent to Vendor Date": "N/A - From RDC" if rdc else po,
              "Scheduled Delivery Date": sched, "Delivered to Client Date": deliv, "Delivery Recorded Date": rec,
              "Product Group": pgroup, "Sub Classification": "Adult", "Vendor": vendor, "Item Description": "x",
              "Line Item Quantity": "100", "Line Item Value": str(value), "Pack Price": "1", "Unit Price": "0.5",
              "First Line Designation": "Yes" if first else "No", "Weight (Kilograms)": weight, "Freight Cost (USD)": freight})
    return r


def synthetic_raw(n_vendors: int = 45, n_ship: int = 1800, seed: int = 11) -> pd.DataFrame:
    """Vendors with persistent late propensities over 2007-2015 plus an internal 'SCMS from RDC' channel."""
    rng = np.random.default_rng(seed)
    vendors = [f"VENDOR {i:02d}" for i in range(n_vendors)]
    p_late = rng.beta(0.8, 10, n_vendors)
    w = rng.pareto(1.1, n_vendors) + 0.05
    rows, lid = [], 1
    for j in range(n_ship):
        internal = rng.random() < 0.35
        v = "SCMS from RDC" if internal else vendors[int(rng.choice(n_vendors, p=w / w.sum()))]
        p = 0.12 if internal else p_late[vendors.index(v)]
        sched = pd.Timestamp("2007-01-01") + pd.Timedelta(days=int(rng.integers(0, 3200)))
        late = rng.random() < p
        off = int(rng.integers(1, 40)) if late else (0 if rng.random() < 0.8 else -int(rng.integers(1, 15)))
        deliv = sched + pd.Timedelta(days=off)
        rec = deliv + pd.Timedelta(days=int(rng.choice([0, 0, 1, 3, 20])))
        country, mode = str(rng.choice(COUNTRIES)), str(rng.choice(MODES, p=[0.7, 0.2, 0.1]))
        n_lines = int(rng.choice([1, 1, 2, 3]))
        fr = round(float(rng.lognormal(7, 1)), 2)
        for k in range(n_lines):
            first = k == 0
            freight = str(fr) if first and rng.random() < 0.7 else ("Freight Included in Commodity Cost" if first else f"See ASN-{j} (ID#:{lid - k})")
            weight = str(round(float(rng.lognormal(5, 1)), 1)) if first else f"See ASN-{j} (ID#:{lid - k})"
            rows.append(_row(lid, f"DN-{j}" if internal else f"ASN-{j}", v, country, mode, str(rng.choice(["ARV", "HRDT"], p=[0.7, 0.3])),
                             fmt(sched), fmt(deliv), fmt(rec), round(float(rng.lognormal(9, 1)), 2), first, freight, weight, rdc=internal))
            lid += 1
    return pd.DataFrame(rows, columns=COLUMNS)


EDGE_CASES = [
    # (case, kwargs)
    ("late5_numeric_freight", dict(ship="ASN-1", vendor="V_A", sched="10-Jan-10", deliv="15-Jan-10", rec="16-Jan-10", value=1000, first=True, freight="100", weight="50")),
    ("same_ship_see_ref", dict(ship="ASN-1", vendor="V_A", sched="10-Jan-10", deliv="15-Jan-10", rec="16-Jan-10", value=500, first=False, freight="See ASN-1 (ID#:1)", weight="See ASN-1 (ID#:1)")),
    ("exact_match_included", dict(ship="ASN-2", vendor="V_A", sched="1-Mar-10", deliv="1-Mar-10", rec="1-Mar-10", value=200, first=True, freight="Freight Included in Commodity Cost", weight="Weight Captured Separately")),
    ("extreme_early_invoiced", dict(ship="ASN-3", vendor="V_B", sched="1-Jun-10", deliv="1-Feb-10", rec="1-Feb-10", value=300, first=True, freight="Invoiced Separately", weight="10")),
    ("severe_late_20", dict(ship="ASN-4", vendor="V_B", sched="1-Jun-11", deliv="21-Jun-11", rec="21-Jun-11", value=400, first=True, freight="40", weight="10")),
    ("late_200_capped", dict(ship="ASN-5", vendor="V_B", sched="1-Jan-11", deliv="20-Jul-11", rec="20-Jul-11", value=100, first=True, freight="10", weight="10")),
    ("internal_rdc", dict(ship="DN-1", vendor="SCMS from RDC", sched="5-May-11", deliv="9-May-11", rec="9-May-11", value=900, first=True, freight="90", weight="90", rdc=True)),
    ("zero_value_zero_weight", dict(ship="ASN-6", vendor="V_A", sched="1-Jan-12", deliv="1-Jan-12", rec="1-Jan-12", value=0, first=True, freight="5", weight="0")),
    ("negative_freight", dict(ship="ASN-7", vendor="V_A", sched="1-Feb-12", deliv="1-Feb-12", rec="1-Feb-12", value=100, first=True, freight="-50", weight="10")),
    ("unparseable_delivered", dict(ship="ASN-8", vendor="V_A", sched="1-Feb-12", deliv="Date Not Captured", rec="1-Feb-12", value=100, first=True, freight="5", weight="10")),
    ("recorded_before_delivered", dict(ship="ASN-9", vendor="V_A", sched="1-Mar-12", deliv="4-Mar-12", rec="1-Mar-12", value=100, first=True, freight="5", weight="10")),
    ("mode_na", dict(ship="ASN-10", vendor="V_A", sched="1-Mar-12", deliv="1-Mar-12", rec="1-Mar-12", value=100, first=True, freight="5", weight="10", mode="N/A")),
    ("freight_ratio_capped", dict(ship="ASN-11", vendor="V_A", sched="1-Apr-12", deliv="1-Apr-12", rec="1-Apr-12", value=100, first=True, freight="5000", weight="10")),
    ("late_by_one_day", dict(ship="ASN-12", vendor="V_A", sched="1-May-12", deliv="2-May-12", rec="2-May-12", value=100, first=True, freight="5", weight="10")),
]


def edge_raw() -> pd.DataFrame:
    rows = []
    for i, (_case, kw) in enumerate(EDGE_CASES, start=1):
        kw = dict(kw)
        rows.append(_row(i, kw.pop("ship"), kw.pop("vendor"), "Zambia", kw.pop("mode", "Air"), "ARV", kw.pop("sched"), kw.pop("deliv"),
                         kw.pop("rec"), kw.pop("value"), kw.pop("first"), kw.pop("freight"), kw.pop("weight"), rdc=kw.pop("rdc", False)))
    return pd.DataFrame(rows, columns=COLUMNS)


def write_edge_fixture() -> Path:
    FIX.mkdir(exist_ok=True)
    out = FIX / "scms_edge_cases.csv"
    e = edge_raw()
    e.insert(0, "case", [c for c, _ in EDGE_CASES])
    e.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    print(write_edge_fixture())
