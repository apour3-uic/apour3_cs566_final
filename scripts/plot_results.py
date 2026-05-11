#!/usr/bin/env python3
"""Plot total (LB + sort) time vs P for each (N, s) combination."""
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
N_VALUES = [100000, 1000000]
S_VALUES = [2, 8]
P_VALUES = [8, 16, 32, 64, 128]

CONFIGS = [
    ("no_lb",   "default", "no LB",              "o-"),
    ("with_lb", "default", "LB (deterministic)", "s-"),
    ("with_lb", "rand",    "LB (random CL)",     "^-"),
]

FIELDS = {
    "Processors":         re.compile(r"^Processors:\s+(\d+)"),
    "Mode":               re.compile(r"^Mode:\s+(\S+)"),
    "CL estimator":       re.compile(r"^CL estimator:\s+(\S+)"),
    "LB time":            re.compile(r"^LB time:\s+([0-9.eE+-]+)"),
    "Sort time":          re.compile(r"^Sort time:\s+([0-9.eE+-]+)"),
    "Final quantitative": re.compile(r"^Final quantitative:\s+([0-9.eE+-]+)"),
    "Final qualitative":  re.compile(r"^Final qualitative:\s+([0-9.eE+-]+)"),
}

def parse_stats(path):
    out = {}
    for line in path.read_text().splitlines():
        for key, pat in FIELDS.items():
            m = pat.match(line)
            if m:
                out[key] = m.group(1)
                break
    return out

def main():
    files = list(OUTPUTS.glob("stats_*.txt"))
    if not files:
        sys.exit(f"No stats files found in {OUTPUTS}")

    # data[(N, s, mode, cl)][P]  = total_time
    # imbal[(N, s, mode, cl)][P] = final_quantitative_imbalance
    # qual[(N, s, mode, cl)][P]  = final_qualitative_imbalance
    data = defaultdict(dict)
    imbal = defaultdict(dict)
    qual = defaultdict(dict)
    for f in files:
        # N and s aren't in the file body — pull from filename
        # filename: stats_<mode>_<...>_<N>_<P>_<s>.txt where final 3 underscore-
        # separated tokens are N, P, s.
        stem = f.stem  # e.g. stats_with_lb_rand_100000_8_2
        parts = stem.split("_")
        try:
            s = int(parts[-1])
            P = int(parts[-2])
            N = int(parts[-3])
        except ValueError:
            continue

        st = parse_stats(f)
        if "LB time" not in st or "Sort time" not in st:
            continue
        mode = st.get("Mode", "")
        cl   = st.get("CL estimator", "")
        total = float(st["LB time"]) + float(st["Sort time"])
        data[(N, s, mode, cl)][P] = total
        if "Final quantitative" in st:
            imbal[(N, s, mode, cl)][P] = float(st["Final quantitative"])
        if "Final qualitative" in st:
            qual[(N, s, mode, cl)][P] = float(st["Final qualitative"])

    # P=1 baseline lives under (N, s=1, no_lb, default)
    baselines = {N: data.get((N, 1, "no_lb", "default"), {}).get(1) for N in N_VALUES}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for row, N in enumerate(N_VALUES):
        for col, s in enumerate(S_VALUES):
            ax = axes[row][col]
            for mode, cl, label, style in CONFIGS:
                series = data.get((N, s, mode, cl), {})
                xs = [P for P in P_VALUES if P in series]
                ys = [series[P] for P in xs]
                if xs:
                    ax.plot(xs, ys, style, label=label)
            if baselines.get(N) is not None:
                ax.axhline(baselines[N], linestyle=":", color="gray",
                           label=f"P=1 ({baselines[N]:.3g} s)")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(P_VALUES)
            ax.set_xticklabels(P_VALUES)
            ax.set_title(f"N = {N:,}, s = {s}")
            ax.set_xlabel("P (processors)")
            ax.set_ylabel("LB + sort time (sec)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

    fig.suptitle("Total time (LB + sort) vs. P", fontsize=14)
    fig.tight_layout()
    out_png = OUTPUTS / "total_time_vs_P.png"
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")

    # Quantitative imbalance plot — same layout, same 3 series. "no LB" is
    # relabeled to "initial" since final == initial when LB is off.
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for row, N in enumerate(N_VALUES):
        for col, s in enumerate(S_VALUES):
            ax = axes2[row][col]
            for mode, cl, label, style in CONFIGS:
                series = imbal.get((N, s, mode, cl), {})
                xs = [P for P in P_VALUES if P in series]
                ys = [series[P] for P in xs]
                if xs:
                    plot_label = "initial" if mode == "no_lb" else label
                    ax.plot(xs, ys, style, label=plot_label)
            ax.set_xscale("log", base=2)
            ax.set_xticks(P_VALUES)
            ax.set_xticklabels(P_VALUES)
            ax.set_title(f"N = {N:,}, s = {s}")
            ax.set_xlabel("P (processors)")
            ax.set_ylabel("Quantitative imbalance")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

    fig2.suptitle("Quantitative imbalance", fontsize=14)
    fig2.tight_layout()
    out_png2 = OUTPUTS / "final_imbalance_vs_P.png"
    fig2.savefig(out_png2, dpi=130)
    print(f"Wrote {out_png2}")

    # Qualitative imbalance plot — same layout, same series, "no LB" → "initial".
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for row, N in enumerate(N_VALUES):
        for col, s in enumerate(S_VALUES):
            ax = axes3[row][col]
            for mode, cl, label, style in CONFIGS:
                series = qual.get((N, s, mode, cl), {})
                xs = [P for P in P_VALUES if P in series]
                ys = [series[P] for P in xs]
                if xs:
                    plot_label = "initial" if mode == "no_lb" else label
                    ax.plot(xs, ys, style, label=plot_label)
            ax.set_xscale("log", base=2)
            ax.set_xticks(P_VALUES)
            ax.set_xticklabels(P_VALUES)
            ax.set_title(f"N = {N:,}, s = {s}")
            ax.set_xlabel("P (processors)")
            ax.set_ylabel("Qualitative imbalance")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

    fig3.suptitle("Qualitative imbalance", fontsize=14)
    fig3.tight_layout()
    out_png3 = OUTPUTS / "final_qual_imbalance_vs_P.png"
    fig3.savefig(out_png3, dpi=130)
    print(f"Wrote {out_png3}")

if __name__ == "__main__":
    main()
