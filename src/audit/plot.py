# src/audit/plot.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RISK_ORDER = [
    "identity_risk",
    "sensitive_risk",
    "exposure_risk",
    "volume_risk",
    "activity_risk",
    "concentration_risk",
]


def plot_radar(row, out_png: Path, title: str = "User Risk Profile"):
    labels = RISK_ORDER
    vals = [float(row[c]) for c in labels]
    # 收尾闭合
    vals = vals + [vals[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_ylim(0, 1)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
