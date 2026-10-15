"""Obtain the USAID / PEPFAR SCMS Delivery History CSV without user credentials, verify it, convert to parquet.

Usage: python scripts/download_data.py [--force] [--source NAME] [--file PATH]

Sources are tried in the order listed in config/config.yaml (data.sources): data.usaid.gov catalogue export, the
original usaid.gov/opengov file, the Internet Archive copy of that original file, then a public GitHub mirror. The
Kaggle copy is used only when KAGGLE_USERNAME / KAGGLE_KEY are set and --source kaggle is requested. A candidate is
accepted only if it parses to the documented shape (10,324 rows, 33 columns, required columns present, unique IDs).
Where config records a SHA-256 for the source, a mismatch is reported (content checks still decide acceptance because
re-saves change bytes, not content).

Output: data/raw/SCMS_Delivery_History_Dataset.csv (canonical UTF-8, all values untouched), data/raw/manifest.json and
data/interim/scms_raw.parquet (every column kept as text; typing/cleaning happens in SQL). Raw data is never committed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_API = "https://www.kaggle.com/api/v1/datasets/download/{ds}"
UA = {"User-Agent": "Mozilla/5.0 (research; supplier-risk-scoring reproducible download)"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str, headers: dict | None = None, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (https URLs from config only)
        return r.read()


def parse_and_verify(raw: bytes, encoding: str, d: dict) -> pd.DataFrame:
    """Parse as text, normalise the header (BOM), and check the documented shape. Raises ValueError otherwise."""
    df = pd.read_csv(io.StringIO(raw.decode(encoding)), dtype=str, keep_default_na=False)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    if len(df) != d["expected_rows"]:
        raise ValueError(f"{len(df):,} rows, expected {d['expected_rows']:,}")
    if df.shape[1] != d["expected_columns"]:
        raise ValueError(f"{df.shape[1]} columns, expected {d['expected_columns']}")
    missing = [c for c in d["required_columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    if df["ID"].nunique() != len(df):
        raise ValueError("ID is not unique")
    return df


def try_kaggle(d: dict) -> tuple[bytes, str] | None:
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return None
    import zipfile
    tok = base64.b64encode(f"{os.environ['KAGGLE_USERNAME']}:{os.environ['KAGGLE_KEY']}".encode()).decode()
    z = zipfile.ZipFile(io.BytesIO(fetch(KAGGLE_API.format(ds=d["kaggle_dataset"]), {"Authorization": f"Basic {tok}"},
                                         300)))
    name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
    return z.read(name), "utf-8-sig"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--source", default="auto", help="auto | kaggle | a source name from config | 'file'")
    ap.add_argument("--file", help="verify and ingest a CSV you downloaded by hand (with --source file)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    d = cfg["data"]
    raw_dir, interim = ROOT / cfg["paths"]["raw_dir"], ROOT / cfg["paths"]["interim_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    target = raw_dir / d["filename"]
    if target.exists() and not args.force:
        df = parse_and_verify(target.read_bytes(), "utf-8", d)
        used = json.loads((raw_dir / "manifest.json").read_text()) if (raw_dir / "manifest.json").exists() else {}
        print(f"Existing {target.name} verified ({len(df):,} rows); use --force to re-download.")
    else:
        attempts, got = [], None
        if args.source == "file":
            raw = Path(args.file).read_bytes()
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    got = (raw, enc, "local_file")
                    parse_and_verify(raw, enc, d)
                    break
                except (UnicodeDecodeError, ValueError):
                    got = None
        else:
            plan = []
            if args.source in ("auto", "kaggle") and os.environ.get("KAGGLE_USERNAME"):
                plan.append({"name": "kaggle", "kaggle": True})
            if args.source != "kaggle":
                plan += [s for s in d["sources"] if args.source in ("auto", s["name"])]
            for s in plan:
                try:
                    if s.get("kaggle"):
                        raw, enc = try_kaggle(d)
                    else:
                        raw, enc = fetch(s["url"]), s["encoding"]
                    parse_and_verify(raw, enc, d)
                    got = (raw, enc, s["name"])
                    if s.get("sha256") and sha256_bytes(raw) != s["sha256"]:
                        print(f"  note: {s['name']} SHA-256 differs from the recorded one (content checks passed)")
                    break
                except Exception as e:  # noqa: BLE001
                    attempts.append(f"{s['name']}: {type(e).__name__}: {str(e)[:120]}")
                    print(f"  FAILED {attempts[-1]}")
        if got is None:
            raise SystemExit("No source could be obtained and verified:\n  " + "\n  ".join(attempts) +
                             "\nDownload the CSV by hand (see data/README.md) and run: "
                             "python scripts/download_data.py --source file --file <path>")
        raw, enc, name = got
        df = parse_and_verify(raw, enc, d)
        df.to_csv(target, index=False, encoding="utf-8")
        used = {"source": name, "source_sha256": sha256_bytes(raw), "source_bytes": len(raw), "source_encoding": enc,
                "failed_sources_before": attempts}
        print(f"Obtained from source '{name}'")
    manifest = {**used, "rows": len(df), "columns": df.shape[1], "canonical_csv_sha256": sha256_bytes(target.read_bytes())}
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    df.to_parquet(interim / "scms_raw.parquet", index=False)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts/data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"OK: {len(df):,} rows x {df.shape[1]} columns -> {target} and data/interim/scms_raw.parquet")


if __name__ == "__main__":
    main()
