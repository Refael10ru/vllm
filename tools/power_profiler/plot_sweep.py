#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot power_sweep.py results: per-phase heatmaps + perf-vs-power pareto.

Usage: python plot_sweep.py results.json [-o outdir]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def heatmap(ax, rows, metric, title):
    sms = sorted({r["sm_mhz"] for r in rows})
    mems = sorted({r["mem_mhz"] for r in rows})
    grid = np.full((len(mems), len(sms)), np.nan)
    for r in rows:
        grid[mems.index(r["mem_mhz"]), sms.index(r["sm_mhz"])] = r[metric]
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(sms)), sms, rotation=45)
    ax.set_yticks(range(len(mems)), mems)
    ax.set_xlabel("SM MHz")
    ax.set_ylabel("Mem MHz")
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(grid):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="w")
    plt.colorbar(im, ax=ax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("-o", "--outdir", default=".")
    args = p.parse_args()
    rows = json.loads(Path(args.results).read_text())
    outdir = Path(args.outdir)
    phases = sorted({r["phase"] for r in rows})

    for r in rows:
        r["tok_s_per_w"] = r["tok_s"] / r["watts_mean"]

    fig, axes = plt.subplots(len(phases), 3, figsize=(16, 5 * len(phases)))
    axes = np.atleast_2d(axes)
    for i, phase in enumerate(phases):
        prows = [r for r in rows if r["phase"] == phase]
        heatmap(axes[i][0], prows, "tok_s", f"{phase}: tok/s")
        heatmap(axes[i][1], prows, "watts_mean", f"{phase}: watts")
        heatmap(axes[i][2], prows, "tok_s_per_w", f"{phase}: tok/s/W")
    fig.suptitle(f"{rows[0]['model']} on {rows[0]['gpu']}")
    fig.tight_layout()
    fig.savefig(outdir / "heatmaps.png", dpi=150)

    fig2, ax = plt.subplots(figsize=(8, 6))
    for phase, marker in zip(phases, "ox^s"):
        prows = [r for r in rows if r["phase"] == phase]
        ax.scatter(
            [r["watts_mean"] for r in prows],
            [r["tok_s"] for r in prows],
            marker=marker,
            label=phase,
        )
        for r in prows:
            ax.annotate(f"{r['sm_mhz']}", (r["watts_mean"], r["tok_s"]), fontsize=6)
    ax.set_xlabel("watts (mean)")
    ax.set_ylabel("tok/s")
    ax.legend()
    ax.set_title("perf vs power (labels = SM MHz)")
    fig2.savefig(outdir / "pareto.png", dpi=150)
    print(f"wrote {outdir}/heatmaps.png, {outdir}/pareto.png")


if __name__ == "__main__":
    main()
