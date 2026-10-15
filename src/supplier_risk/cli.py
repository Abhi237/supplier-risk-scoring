"""Command-line interface: `supplier-risk <command>`."""
from __future__ import annotations

import json
import subprocess
import sys

import click

from . import pipeline
from .config import ROOT


def _p(obj):
    click.echo(json.dumps(obj, indent=2, default=str)[:6000])


@click.group()
def main():
    """Vendor delivery-risk pipeline (USAID / PEPFAR SCMS vendors)."""


@main.command()
def data():
    """Obtain the SCMS Delivery History CSV from public sources, verify it, convert to parquet."""
    subprocess.run([sys.executable, str(ROOT / "scripts/download_data.py")], check=True)


@main.command()
def sql():
    """Run numbered DuckDB SQL scripts (staging, cleaning log, line intervals, vendor summary, panel, OOT)."""
    _p(pipeline.run_sql())


@main.command()
def train():
    """Shrunk vendor scores + bootstrap intervals, archetypes, OOT validation, exposure and scenarios."""
    _p(pipeline.train())


@main.command(name="all")
@click.pass_context
def all_(ctx):
    """Run sql -> train."""
    for cmd in (sql, train):
        ctx.invoke(cmd)


if __name__ == "__main__":
    main()
