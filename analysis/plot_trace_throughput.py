#!/usr/bin/env python3
"""
Plot DualState trace-driven throughput over time.
Similar to ppopp/plot_fig9.py but for DualState vs SGLang baseline.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 1.15, "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
    "lines.linewidth": 2.35, "lines.markersize": 5.7,
    "figure.dpi": 150, "savefig.dpi": 600,
})

METHOD_STYLE = {
    "SGLang P/D": {"color": "#d62728", "marker": "o", "mfc": "white"},
    "DualState":  {"color": "#1f77b4", "marker": "D", "mfc": "#1f77b4"},
}


def plot(data_path, output_dir):
    data = json.load(open(data_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = list(data.keys())
    fig, axes = plt.subplots(1, len(traces), figsize=(3.7 * len(traces), 2.82), sharey=True)
    if len(traces) == 1:
        axes = [axes]

    for ax, trace_name in zip(axes, traces):
        trace = data[trace_name]
        minutes = sorted(int(k) for k in next(iter(trace.values())).keys())
        time_min = np.array(minutes)

        for method, style in METHOD_STYLE.items():
            if method not in trace:
                continue
            y = np.array([trace[method].get(str(m), {}).get("throughput", 0) for m in minutes])
            ax.plot(time_min, y, label=method,
                    color=style["color"], marker=style["marker"],
                    markerfacecolor=style["mfc"], markeredgecolor=style["color"],
                    markeredgewidth=1.45, markevery=max(1, len(minutes)//10))
            ax.fill_between(time_min, 0, y, color=style["color"], alpha=0.07)

        # TTFT subplot (secondary axis)
        ax2 = ax.twinx()
        for method, style in METHOD_STYLE.items():
            if method not in trace:
                continue
            ttft = np.array([trace[method].get(str(m), {}).get("ttft_mean", 0) * 1000 for m in minutes])
            ax2.plot(time_min, ttft, linestyle="--", alpha=0.5,
                     color=style["color"], linewidth=1.5)
        ax2.set_ylabel("TTFT (ms)", fontsize=11, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=10)

        ax.set_title(trace_name, pad=7)
        ax.set_xlabel("Time (min)")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.42)
        ax.spines["top"].set_visible(False)

    axes[0].set_ylabel("Throughput (req/min)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.065), ncol=len(METHOD_STYLE),
               frameon=False, columnspacing=1.8)

    trace_labels = ["(a) Kimi K25 trace", "(b) Qwen-Bailian trace"]
    for i, ax in enumerate(axes):
        if i < len(trace_labels):
            fig.text((i + 0.5) / len(traces), -0.03, trace_labels[i],
                     ha="center", va="top", fontsize=14)

    fig.subplots_adjust(left=0.1, right=0.92, bottom=0.235, top=0.785, wspace=0.35)

    fig.savefig(output_dir / "dualstate_trace_throughput.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "dualstate_trace_throughput.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'dualstate_trace_throughput.pdf'}")

    # Print summary
    for trace_name in traces:
        trace = data[trace_name]
        for method in METHOD_STYLE:
            if method not in trace:
                continue
            vals = [v.get("throughput", 0) for v in trace[method].values()]
            ttfts = [v.get("ttft_mean", 0) * 1000 for v in trace[method].values() if v.get("ttft_mean", 0) > 0]
            if vals:
                print(f"  {trace_name} {method}: avg_tput={np.mean(vals):.2f} req/min, "
                      f"avg_ttft={np.mean(ttfts):.0f}ms" if ttfts else "")


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "plot_data.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    plot(data_path, output_dir)
