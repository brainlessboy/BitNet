#!/usr/bin/env python3
"""
Plot training curves from BitDistill training log.

Usage:
    python distill/plot.py                                  # default log path
    python distill/plot.py distill/training_log.jsonl       # custom log path
    python distill/plot.py -o training_curves.png           # custom output path
"""

import argparse
import json
import sys


def load_log(path):
    total_steps = 0
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") == "header":
                total_steps = d.get("total_steps", 0)
            elif d.get("type") == "step":
                records.append(d)
    return total_steps, records


def main():
    parser = argparse.ArgumentParser(description="Plot BitDistill training curves")
    parser.add_argument("log_file", nargs="?", default="distill/training_log.jsonl",
                        help="Path to training_log.jsonl")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG path (default: same dir as log file)")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        sys.exit(1)

    total_steps, records = load_log(args.log_file)
    if not records:
        print(f"No training data found in {args.log_file}")
        sys.exit(1)

    steps = [r["step"] for r in records]

    output = args.output
    if output is None:
        import os
        output = os.path.join(os.path.dirname(args.log_file), "training_plot.png")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    charts = [
        ("ce", "CE (cross-entropy)", "#00bcd4"),
        ("ld", "LD (logit distillation)", "#ff9800"),
        ("ad", "AD (attention distillation)", "#e91e63"),
        ("loss", "Total Loss", "#4caf50"),
    ]

    for ax, (key, label, color) in zip(axes, charts):
        vals = [r[key] for r in records]
        ax.plot(steps, vals, color=color, linewidth=1.2)
        ax.fill_between(steps, vals, alpha=0.1, color=color)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=9)
        # Show current value
        ax.annotate(f"{vals[-1]:.4g}", xy=(steps[-1], vals[-1]),
                    fontsize=8, color=color, fontweight="bold",
                    xytext=(5, 0), textcoords="offset points")

    axes[-1].set_xlabel("Step", fontsize=10)

    title = f"BitDistill Training — step {steps[-1]}"
    if total_steps:
        title += f"/{total_steps}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
