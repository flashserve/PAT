#!/usr/bin/env python3
"""
python plot_bench.py --root /path/to/e2e_results.log --out fig/eval_e2e_overall_p99.pdf
"""
import argparse
import os
import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

BACKEND_STYLE_MAP = {
    "PREFIX_ATTN": ("PAT", "*", 0),
    "Relay_ATTN": ("RelayAttention++", "x", 1),
    "FLASH_ATTN": ("FlashAttention", ".", 2),
    "FLASHINFER": ("FlashInfer", "+", 3),
}

METRICS = ["TTFT", "TPOT", "P99"]


def _format_title_from_key(key: str) -> str:
    if not key:
        return ""
    if "-" in key:
        parts = key.split("-")
        dataset = parts[-1]
        model = "-".join(parts[:-1])
    else:
        model, dataset = key, ""

    dataset = re.sub(r"\bburst\b", "conversation", dataset, flags=re.I)
    return f"{model}  {dataset}".strip()


def extract_data(root: str):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    files = []
    if os.path.isfile(root):
        files.append(root)
    elif os.path.isdir(root):
        import glob

        files.extend(glob.glob(os.path.join(root, "*.jsonl")))
        files.extend(glob.glob(os.path.join(root, "*.log")))

    print(f"Loading data from: {files}")

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except:
                    continue

                if record.get("status") != "success":
                    continue

                cfg = record.get("config", {})
                met = record.get("metrics", {})

                backend_raw = cfg.get("ATTENTION_BACKEND", "")
                if backend_raw not in BACKEND_STYLE_MAP:
                    continue

                display_name = BACKEND_STYLE_MAP[backend_raw][0]

                full_model = cfg.get("model", "")
                model = full_model.split("/")[-1] if "/" in full_model else full_model
                dataset = cfg.get("trace", "")
                key = f"{model}-{dataset}"

                try:
                    qps = float(cfg.get("request_rate", 0))
                except:
                    continue

                val_ttft = met.get("mean_ttft_ms")
                val_tpot = met.get("mean_tpot_ms")
                val_p99 = met.get("p99_tpot_ms")

                if val_ttft is not None:
                    data[key]["TTFT"][display_name].append((qps, float(val_ttft)))
                if val_tpot is not None:
                    data[key]["TPOT"][display_name].append((qps, float(val_tpot)))
                if val_p99 is not None:
                    data[key]["P99"][display_name].append((qps, float(val_p99)))

    for key in data:
        for metric in data[key]:
            for fname in data[key][metric]:
                data[key][metric][fname].sort(key=lambda x: x[0])
    return data


def plot_grid(data, out_path="figure.pdf", log_scale=True):
    keys = sorted(data.keys())
    keys = (keys + [""] * (4 - len(keys)))[:4]

    fig, axes = plt.subplots(3, 4, figsize=(12, 3.8), sharex=False, sharey=False)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax in axes.flat:
        ax.grid(True, ls="--", alpha=0.5, axis="y", which="major")
        if log_scale:
            ax.grid(True, ls=":", alpha=0.3, axis="y", which="minor")

        for sp in ax.spines.values():
            sp.set_alpha(0.4)
        ax.tick_params(axis="both", length=0)

    style_lookup = {}
    for raw, (disp, mk, c_idx) in BACKEND_STYLE_MAP.items():
        style_lookup[disp] = (mk, cols[c_idx % len(cols)])

    for c, key in enumerate(keys):
        for r, metric in enumerate(METRICS):
            ax = axes[r, c]
            if key and metric in data[key]:
                defined_backends = [v[0] for v in BACKEND_STYLE_MAP.values()]

                for disp_name in defined_backends:
                    series = data[key][metric].get(disp_name, [])
                    if not series:
                        continue

                    xs, ys = zip(*series)
                    final_ys = [y / 1000.0 if metric == "TTFT" else y for y in ys]

                    marker, color = style_lookup.get(disp_name, ("o", "black"))

                    ax.plot(
                        xs,
                        final_ys,
                        label=disp_name,
                        marker=marker,
                        linestyle="-",
                        linewidth=1.5,
                        markersize=6,
                        color=color,
                    )

            if log_scale:
                ax.set_yscale("log")

            if r == 0:
                ax.set_title(_format_title_from_key(key), fontsize=11)
            if r == 2:
                ax.set_xlabel("request rate (reqs/s)")
            if r != 2:
                ax.set_xticks([])

    axes[0, 0].set_ylabel("TTFT (s)")
    axes[1, 0].set_ylabel("TPOT (ms)")
    axes[2, 0].set_ylabel("P99 TPOT (ms)")

    row_tags = ["a", "b", "c"]
    for r in range(3):
        for c in range(4):
            ax = axes[r, c]
            tag = f"{row_tags[r]}{c + 1}"
            ax.text(
                0.95,
                0.1,
                tag,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                color="purple",
                bbox=dict(boxstyle="circle,pad=0.15", fc="none", ec="purple", lw=1),
                zorder=10,
                clip_on=False,
            )

    proxies = []
    labels = []
    for _, (disp, mk, c_idx) in BACKEND_STYLE_MAP.items():
        color = cols[c_idx % len(cols)]
        proxies.append(
            Line2D([0], [0], color=color, marker=mk, linewidth=1.8, markersize=4.3)
        )
        labels.append(disp)

    fig.legend(
        handles=proxies,
        labels=labels,
        loc="upper center",
        ncol=len(proxies),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 1),
        borderaxespad=0.0,
        handlelength=2.0,
    )

    fig.subplots_adjust(
        left=0.05, right=0.995, bottom=0.1, top=0.88, wspace=0.15, hspace=0.075
    )

    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        type=str,
        default="e2e_perf.jsonl",
        help="Path to the JSONL log file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="fig/eval_e2e_overall_p99.pdf",
        help="Output PDF path",
    )
    parser.add_argument(
        "--no-log", action="store_true", help="Disable log scale (default is enabled)"
    )

    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    use_log_scale = not args.no_log

    data = extract_data(args.log_file)
    plot_grid(data, args.out, log_scale=use_log_scale)
    print(f"[OK] Saved: {args.out} (Log Scale: {use_log_scale})")


if __name__ == "__main__":
    main()
