"""Charts saved as PNG files."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd

COLOR = "#2a6f97"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_monthly(monthly: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = monthly["month"].astype(str)
    ax.plot(x, monthly["revenue"], marker="o", color=COLOR, linewidth=2)
    ax.set_title("Monthly revenue")
    ax.set_ylabel("€")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


def plot_top_products(top: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    data = top.sort_values("revenue")
    ax.barh(data["product"].astype(str), data["revenue"], color=COLOR)
    ax.set_title("Top products by revenue")
    ax.xaxis.set_major_formatter(lambda v, _: f"€{v/1000:.0f}k")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


def plot_region_share(share: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    data = share.sort_values("share_pct")
    bars = ax.barh(data["region"].astype(str), data["share_pct"], color=COLOR)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_title("Revenue share by region")
    ax.set_xlabel("%")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)
