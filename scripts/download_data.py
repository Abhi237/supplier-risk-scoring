"""Download the Olist Brazilian E-Commerce CSVs from a public mirror, verify integrity, convert to parquet.

Usage: python scripts/download_data.py [--force]
Kaggle requires login, so a public GitHub mirror of the original CSVs is used (see data/README.md).
Integrity: row counts are checked against the documented dataset sizes and SHA-256 digests are recorded.
Raw data is never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
KEY = {"olist_orders_dataset.csv": "orders", "olist_order_items_dataset.csv": "order_items",
       "olist_sellers_dataset.csv": "sellers"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    raw, interim = ROOT / cfg["paths"]["raw_dir"], ROOT / cfg["paths"]["interim_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for f in cfg["data"]["files"]:
        dst = raw / f
        if args.force or not dst.exists():
            url = f"{cfg['data']['mirror_base']}/{f}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, dst)
        df = pd.read_csv(dst)
        manifest[f] = {"rows": len(df), "sha256": sha256(dst), "bytes": dst.stat().st_size}
        if f in KEY:
            exp = cfg["data"]["expected_rows"][KEY[f]]
            if len(df) != exp:
                raise SystemExit(f"Integrity check failed: {f} has {len(df):,} rows, expected {exp:,}")
        df.to_parquet(interim / f.replace(".csv", ".parquet"), index=False)
        print(f"  {f}: {len(df):,} rows OK")
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts/data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Integrity verified; manifest written.")


if __name__ == "__main__":
    main()
