#!/usr/bin/env python3
"""
Live training dashboard for BitDistill.

Reads the JSONL training log and renders a full-color terminal dashboard.
Auto-refreshes as new data arrives.

Usage:
    python distill/dashboard.py distill/training_log.jsonl
"""

import argparse
import curses
import json
import os
import time


def load_log(path):
    """Load training log, returning (total_steps, records)."""
    total_steps = 0
    records = []
    try:
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
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return total_steps, records


def fmt_time(seconds):
    if seconds <= 0:
        return "--"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def draw_chart(lines, y, x, width, height, data, label, color_pair, stdscr):
    """Draw a line chart. Returns number of rows used."""
    if not data:
        stdscr.addstr(y, x, f"  {label}: waiting for data...", curses.A_DIM)
        return 3

    # Fit all data into available chart width by downsampling
    chart_w = width - 14
    if chart_w < 10:
        chart_w = 10

    if len(data) <= chart_w:
        plot_data = list(data)
    else:
        # Bucket all data points into chart_w bins, average each bucket
        plot_data = []
        bucket_size = len(data) / chart_w
        for i in range(chart_w):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            bucket = data[start:end]
            if bucket:
                plot_data.append(sum(bucket) / len(bucket))
            else:
                plot_data.append(data[-1])

    lo = min(plot_data)
    hi = max(plot_data)
    if hi == lo:
        hi = lo + 1.0

    # Title line
    current = plot_data[-1]
    title = f" {label}"
    try:
        stdscr.addstr(y, x, title, curses.color_pair(color_pair) | curses.A_BOLD)
        stdscr.addstr(f"  {current:.4g}", curses.A_NORMAL)
    except curses.error:
        pass
    y += 1

    # Chart rows
    for row in range(height - 1, -1, -1):
        # Y-axis label
        if row == height - 1:
            y_label = f"{hi:>10.3g}"
        elif row == 0:
            y_label = f"{lo:>10.3g}"
        elif row == (height - 1) // 2:
            mid = lo + (hi - lo) * 0.5
            y_label = f"{mid:>10.3g}"
        else:
            y_label = " " * 10

        line = y_label + " │"
        try:
            stdscr.addstr(y, x, line, curses.A_DIM)
        except curses.error:
            pass

        # Plot points
        col_start = x + len(line)
        for i, v in enumerate(plot_data):
            normalized = (v - lo) / (hi - lo) * (height - 1)
            if col_start + i >= width + x:
                break
            try:
                if round(normalized) == row:
                    # This is the data point
                    if row > 0:
                        stdscr.addstr(y, col_start + i, "●",
                                      curses.color_pair(color_pair) | curses.A_BOLD)
                    else:
                        stdscr.addstr(y, col_start + i, "●",
                                      curses.color_pair(color_pair) | curses.A_BOLD)
                elif normalized > row and row > 0:
                    stdscr.addstr(y, col_start + i, "│",
                                  curses.color_pair(color_pair) | curses.A_DIM)
                elif row == 0:
                    stdscr.addstr(y, col_start + i, "─", curses.A_DIM)
                else:
                    stdscr.addstr(y, col_start + i, " ")
            except curses.error:
                pass
        y += 1

    # X-axis
    axis = " " * 10 + " └" + "─" * len(plot_data)
    try:
        stdscr.addstr(y, x, axis[:width], curses.A_DIM)
    except curses.error:
        pass
    y += 1

    # Step labels
    if len(data) > 1:
        first = data[max(0, len(data) - chart_w)].get("step", 0) if isinstance(data[0], dict) else 0
        last_step = data[-1].get("step", 0) if isinstance(data[0], dict) else len(data)
        # For raw value lists, use indices from records
        step_line = " " * 12
        try:
            stdscr.addstr(y, x, step_line, curses.A_DIM)
        except curses.error:
            pass
    y += 1

    return y


def draw_progress_bar(stdscr, y, x, width, step, total_steps, elapsed, eta):
    """Draw a progress bar."""
    bar_width = min(width - 40, 50)
    if bar_width < 10:
        bar_width = 10

    frac = step / max(total_steps, 1)
    filled = int(bar_width * frac)

    try:
        stdscr.addstr(y, x, "  ")
        # Filled portion
        for i in range(filled):
            stdscr.addstr("█", curses.color_pair(4) | curses.A_BOLD)
        # Empty portion
        for i in range(bar_width - filled):
            stdscr.addstr("░", curses.A_DIM)

        info = f"  {step}/{total_steps}  {frac*100:.1f}%"
        info += f"  elapsed {fmt_time(elapsed)}"
        info += f"  ETA {fmt_time(eta)}"
        stdscr.addstr(info)
    except curses.error:
        pass


def draw_dashboard(stdscr, path):
    """Main curses drawing loop."""
    curses.curs_set(0)  # hide cursor
    stdscr.timeout(1000)  # refresh every 1s

    # Colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)     # CE
    curses.init_pair(2, curses.COLOR_YELLOW, -1)    # LD
    curses.init_pair(3, curses.COLOR_GREEN, -1)     # Total loss
    curses.init_pair(4, curses.COLOR_BLUE, -1)      # progress bar
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)   # header
    curses.init_pair(6, curses.COLOR_RED, -1)       # AD
    curses.init_pair(7, curses.COLOR_WHITE, -1)     # dim text

    last_mtime = 0
    total_steps = 0
    records = []

    while True:
        # Check for file changes
        try:
            mtime = os.path.getmtime(path)
        except FileNotFoundError:
            mtime = 0

        if mtime != last_mtime:
            total_steps, records = load_log(path)
            last_mtime = mtime

        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()

        # Header
        header = " BitDistill Training Dashboard "
        try:
            pad = max(0, (max_x - len(header)) // 2)
            stdscr.addstr(0, 0, "═" * max_x, curses.color_pair(5))
            stdscr.addstr(1, pad, header, curses.color_pair(5) | curses.A_BOLD)
            stdscr.addstr(2, 0, "═" * max_x, curses.color_pair(5))
        except curses.error:
            pass

        y = 4

        if not records:
            try:
                if os.path.exists(path):
                    stdscr.addstr(y, 2, "Waiting for first training step...",
                                  curses.A_DIM)
                else:
                    stdscr.addstr(y, 2, f"Waiting for log file: {path}",
                                  curses.A_DIM)
                stdscr.addstr(y + 2, 2, "Press 'q' to quit", curses.A_DIM)
            except curses.error:
                pass
        else:
            h = records[-1]

            # Summary line
            try:
                stdscr.addstr(y, 2, f"Step ", curses.A_DIM)
                stdscr.addstr(f"{h['step']}/{total_steps}",
                              curses.color_pair(3) | curses.A_BOLD)
                stdscr.addstr(f"  │  loss ", curses.A_DIM)
                stdscr.addstr(f"{h['loss']:.4f}", curses.color_pair(2) | curses.A_BOLD)
                stdscr.addstr(f"  │  CE ", curses.A_DIM)
                stdscr.addstr(f"{h['ce']:.4f}", curses.color_pair(1))
                stdscr.addstr(f"  LD ", curses.A_DIM)
                stdscr.addstr(f"{h['ld']:.1f}", curses.color_pair(2))
                stdscr.addstr(f"  AD ", curses.A_DIM)
                stdscr.addstr(f"{h['ad']:.2f}", curses.color_pair(6))
                stdscr.addstr(f"  │  lr ", curses.A_DIM)
                stdscr.addstr(f"{h['lr']:.2e}", curses.color_pair(5))
            except curses.error:
                pass

            y += 2

            # Determine chart height based on terminal size
            # 3 charts + summary + progress bar + header
            available = max_y - y - 6
            chart_h = max(4, available // 3 - 3)

            # CE chart
            ce_vals = [r["ce"] for r in records]
            draw_chart([], y, 0, max_x, chart_h, ce_vals,
                       "CE (cross-entropy)", 1, stdscr)
            y += chart_h + 3

            # LD chart
            ld_vals = [r["ld"] for r in records]
            draw_chart([], y, 0, max_x, chart_h, ld_vals,
                       "LD (logit distillation)", 2, stdscr)
            y += chart_h + 3

            # Total loss chart
            loss_vals = [r["loss"] for r in records]
            draw_chart([], y, 0, max_x, chart_h, loss_vals,
                       "Total loss", 3, stdscr)
            y += chart_h + 3

            # Progress bar at bottom
            prog_y = max_y - 3
            try:
                stdscr.addstr(prog_y - 1, 0, "─" * max_x, curses.A_DIM)
            except curses.error:
                pass
            draw_progress_bar(stdscr, prog_y, 0, max_x,
                              h["step"], total_steps,
                              h["elapsed"], h["eta"])
            try:
                stdscr.addstr(prog_y + 1, 0, "─" * max_x, curses.A_DIM)
            except curses.error:
                pass

        # Footer
        try:
            stdscr.addstr(max_y - 1, 2, " q: quit  ", curses.A_DIM)
        except curses.error:
            pass

        stdscr.refresh()

        # Key handling
        key = stdscr.getch()
        if key == ord("q") or key == ord("Q") or key == 27:  # q or ESC
            break


def main():
    parser = argparse.ArgumentParser(description="BitDistill training dashboard")
    parser.add_argument("log_file", nargs="?", default="distill/training_log.jsonl",
                        help="Path to training_log.jsonl")
    args = parser.parse_args()

    curses.wrapper(draw_dashboard, args.log_file)


if __name__ == "__main__":
    main()
