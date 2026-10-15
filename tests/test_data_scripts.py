"""download_data verification logic and the repository hygiene rules (no raw data, no emoji, no em dash)."""
import importlib.util
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts" / "download_data.py")
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)
D = {"expected_rows": 3, "expected_columns": 2, "required_columns": ["ID", "Vendor"]}


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def test_verify_accepts_documented_shape():
    df = pd.DataFrame({"ID": ["1", "2", "3"], "Vendor": list("abc")})
    assert len(dd.parse_and_verify(csv_bytes(df), "utf-8", D)) == 3


def test_verify_rejects_wrong_row_count_missing_column_and_duplicate_ids():
    with pytest.raises(ValueError, match="rows"):
        dd.parse_and_verify(csv_bytes(pd.DataFrame({"ID": ["1"], "Vendor": ["a"]})), "utf-8", D)
    with pytest.raises(ValueError, match="missing"):
        dd.parse_and_verify(csv_bytes(pd.DataFrame({"ID": ["1", "2", "3"], "Other": list("abc")})), "utf-8", D)
    with pytest.raises(ValueError, match="unique"):
        dd.parse_and_verify(csv_bytes(pd.DataFrame({"ID": ["1", "1", "3"], "Vendor": list("abc")})), "utf-8", D)


def test_verify_strips_bom_and_keeps_text_untouched():
    raw = "\ufeffID,Vendor\n1,a\n2,b\n3,c\n".encode("utf-8")
    df = dd.parse_and_verify(raw, "utf-8", D)
    assert list(df.columns) == ["ID", "Vendor"] and df.ID.tolist() == ["1", "2", "3"]


def test_fixture_has_real_column_layout(edge_raw_df, cfg):
    assert edge_raw_df.shape[1] == cfg["data"]["expected_columns"]
    assert set(cfg["data"]["required_columns"]) <= set(edge_raw_df.columns)


def test_raw_data_is_gitignored():
    out = subprocess.run(["git", "check-ignore", "data/raw/SCMS_Delivery_History_Dataset.csv", "data/interim/scms_raw.parquet"],
                         cwd=ROOT, capture_output=True, text=True)
    assert len(out.stdout.split()) == 2


def test_no_raw_line_level_file_is_tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    assert not [f for f in out if f.startswith("data/raw") and not f.endswith(".gitkeep")]
    assert not [f for f in out if f.endswith("lines_for_r.csv") or f.endswith("fct_lines.parquet")]


def _tracked_text():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    for f in out:
        if f.endswith((".py", ".md", ".sql", ".yaml", ".yml", ".R", ".qmd", ".toml", ".txt", ".csv", ".json", "Makefile")) and (ROOT / f).is_file():
            if f.startswith("tests/fixtures") or f in ("renv.lock",):
                continue
            yield f, (ROOT / f).read_text(errors="ignore")


def test_no_emoji_or_em_dash_in_repo_text():
    bad = []
    for f, t in _tracked_text():
        for ch in t:
            o = ord(ch)
            if o in (0x2014, 0x2013) or 0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or o == 0xFE0F:
                bad.append((f, hex(o)))
                break
    assert not bad, bad
