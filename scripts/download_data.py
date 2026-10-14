"""Download the Olist Brazilian E-Commerce CSVs, verify integrity, convert to parquet.

Usage: python scripts/download_data.py [--force] [--source {auto,kaggle,mirror}]

Primary source is Kaggle (olistbr/brazilian-ecommerce). With KAGGLE_USERNAME and KAGGLE_KEY in the environment
the dataset zip is fetched from the Kaggle API (or via the `kaggle` CLI if installed); otherwise the script falls
back to the verified public GitHub mirror in config.yaml. Either path is verified against the row counts and
SHA-256 digests recorded in config.yaml (data.recorded): row counts are a hard check; a SHA-256 mismatch aborts on
the mirror path and is reported as a warning on the Kaggle path (re-packaging can change line endings/bytes
without changing content). Raw data is never committed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_API = "https://www.kaggle.com/api/v1/datasets/download/{ds}"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kaggle_available() -> bool:
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def fetch_kaggle(ds: str, files: list[str], raw: Path) -> None:
    if shutil.which("kaggle"):
        subprocess.run(["kaggle", "datasets", "download", "-d", ds, "-p", str(raw), "--unzip", "--force"], check=True)
        return
    tok = base64.b64encode(f"{os.environ['KAGGLE_USERNAME']}:{os.environ['KAGGLE_KEY']}".encode()).decode()
    req = urllib.request.Request(KAGGLE_API.format(ds=ds), headers={"Authorization": f"Basic {tok}"})
    with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 (fixed https host)
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    for f in files:
        (raw / f).write_bytes(zf.read(f))


def fetch_mirror(base: str, files: list[str], raw: Path, force: bool) -> None:
    for f in files:
        dst = raw / f
        if force or not dst.exists():
            print(f"Downloading {base}/{f}")
            urllib.request.urlretrieve(f"{base}/{f}", dst)  # noqa: S310


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--source", choices=["auto", "kaggle", "mirror"], default="auto")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    d = cfg["data"]
    raw, interim = ROOT / cfg["paths"]["raw_dir"], ROOT / cfg["paths"]["interim_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    have_all = all((raw / f).exists() for f in d["files"])
    source = args.source
    if source == "auto":
        source = "kaggle" if kaggle_available() else "mirror"
    if source == "kaggle" and not kaggle_available():
        raise SystemExit("--source kaggle needs KAGGLE_USERNAME and KAGGLE_KEY")
    if args.force or not have_all:
        if source == "kaggle":
            print(f"Downloading Kaggle dataset {d['kaggle_dataset']} (primary source)")
            try:
                fetch_kaggle(d["kaggle_dataset"], d["files"], raw)
            except Exception as e:  # noqa: BLE001
                if args.source == "kaggle":
                    raise
                print(f"  Kaggle download failed ({e}); falling back to the verified mirror")
                source = "mirror"
        if source == "mirror":
            print("Using verified public mirror (set KAGGLE_USERNAME/KAGGLE_KEY to use Kaggle)")
            fetch_mirror(d["mirror_base"], d["files"], raw, args.force)
    else:
        source = "existing files in data/raw"
    manifest = {"_source": source}
    for f in d["files"]:
        dst = raw / f
        df = pd.read_csv(dst)
        digest = sha256(dst)
        rec = d["recorded"][f]
        manifest[f] = {"rows": len(df), "sha256": digest, "bytes": dst.stat().st_size,
                       "sha256_matches_recorded": digest == rec["sha256"]}
        if len(df) != rec["rows"]:
            raise SystemExit(f"Integrity check failed: {f} has {len(df):,} rows, recorded {rec['rows']:,}")
        if digest != rec["sha256"]:
            msg = f"{f}: SHA-256 {digest[:12]}.. differs from recorded {rec['sha256'][:12]}.."
            if source == "mirror":
                raise SystemExit(f"Integrity check failed: {msg}")
            print(f"  WARNING {msg} (row count matches)")
        df.to_parquet(interim / f.replace(".csv", ".parquet"), index=False)
        print(f"  {f}: {len(df):,} rows OK")
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts/data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Integrity verified (source: {source}); manifest written.")


if __name__ == "__main__":
    main()
