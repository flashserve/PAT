import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def _safe_stats(xs: List[float]) -> Tuple[float, float, float]:
    """Return (mean, min, max). xs must be non-empty."""
    mean_v = sum(xs) / len(xs)
    return mean_v, min(xs), max(xs)


def _format_pct(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}%"


def _format_x(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}x"


def summarize_kernel_perf(
    log_file: str,
    pat_key: str = "pat",
    baseline_keys: Optional[List[str]] = None,
    tree_filter: Optional[str] = None,
    digits_pct: int = 2,
    digits_ratio: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Summarize kernel latency stats.

    Outputs:
      (1) For each baseline: PAT vs baseline reduction in kernel latency (%): mean/min/max over all (tree, head) configs.
      (2) Over all baselines and all (tree, head) configs: (baseline latency / PAT latency) ratio: mean/min/max.
      (3) If tree_filter is provided: same as (1) but only within the specified tree.

    Definition:
      reduction(%) = (baseline - pat) / baseline * 100
      ratio(x)     = baseline / pat

    Returns:
      A dict with raw numeric stats (useful if you want to print elsewhere).
    """
    with open(log_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Discover baselines if not specified.
    if baseline_keys is None:
        keys = set()
        for e in raw_data:
            lat = e.get("latencies", {}) or {}
            keys.update(lat.keys())
        keys.discard(pat_key)
        baseline_keys = sorted(keys)

    # Collect lists
    red_all: Dict[str, List[float]] = defaultdict(list)    # baseline -> [reduction%...]
    red_tree: Dict[str, List[float]] = defaultdict(list)   # baseline -> [reduction%...] within tree_filter
    ratios_all: Dict[str, List[float]] = defaultdict(list)  # baseline -> [baseline/pat...]
    
    skipped = 0
    used = 0

    for e in raw_data:
        tree = e.get("tree", None)
        lat = e.get("latencies", {}) or {}
        if pat_key not in lat:
            skipped += 1
            continue
        pat_lat = lat.get(pat_key, None)
        if pat_lat is None or pat_lat <= 0:
            skipped += 1
            continue

        used += 1
        tree = e.get("tree", None)

        for b in baseline_keys:
            b_lat = lat.get(b, None)
            if b_lat is None or b_lat <= 0:
                continue

            # (1) reduction (%)
            if tree not in ['256_1024', '256_4096']:
            # if True:
                reduction = (b_lat - pat_lat) / b_lat * 100.0
                red_all[b].append(reduction)

            # (2) ratio baseline / pat
            ratios_all[b].append(b_lat / pat_lat)

            # (3) tree-filtered reduction (%)
            if tree_filter is not None and tree == tree_filter:
                red_tree[b].append(reduction)

    # ---- Print formatted outputs ----
    header = f"Kernel Perf Summary (log_file={log_file})"
    if tree_filter is not None:
        header += f", tree_filter={tree_filter}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print(f"- PAT key: {pat_key}")
    print(f"- Baselines: {baseline_keys}")
    print(f"- Entries used (with valid PAT latency): {used}, skipped: {skipped}\n")

    # (1) PAT vs each baseline reduction stats
    print("[1] PAT vs Baseline: kernel latency reduction (%) over trees [1-18] & all head configs")
    print(f"{'Baseline':<12}  {'Mean':>10}  {'Min':>10}  {'Max':>10}  {'N':>6}")
    print("-" * 56)
    stats_out: Dict[str, Dict[str, float]] = {"reduction_all": {}, "ratio_all": {}}

    for b in baseline_keys:
        xs = red_all.get(b, [])
        if not xs:
            print(f"{b:<12}  {'NA':>10}  {'NA':>10}  {'NA':>10}  {0:>6}")
            continue
        mean_v, min_v, max_v = _safe_stats(xs)
        print(
            f"{b:<12}  {_format_pct(mean_v, digits_pct):>10}  "
            f"{_format_pct(min_v, digits_pct):>10}  {_format_pct(max_v, digits_pct):>10}  {len(xs):>6}"
        )
        stats_out["reduction_all"][b] = {"mean": mean_v, "min": min_v, "max": max_v, "n": len(xs)}

    print("")

    # (2) baseline / PAT ratios per baseline over all configs
    print("[2] Baseline/PAT ratio (x) over ALL trees & head configs (per baseline)")
    print(f"{'Baseline':<12}  {'Mean':>10}  {'Min':>10}  {'Max':>10}  {'N':>6}")
    print("-" * 56)

    stats_out["ratio_all"] = {}

    for b in baseline_keys:
        xs = ratios_all.get(b, [])
        if not xs:
            print(f"{b:<12}  {'NA':>10}  {'NA':>10}  {'NA':>10}  {0:>6}")
            continue

        mean_r, min_r, max_r = _safe_stats(xs)
        print(
            f"{b:<12}  {_format_x(mean_r, digits_ratio):>10}  "
            f"{_format_x(min_r, digits_ratio):>10}  {_format_x(max_r, digits_ratio):>10}  {len(xs):>6}"
        )
        stats_out["ratio_all"][b] = {"mean": mean_r, "min": min_r, "max": max_r, "n": len(xs)}

    print("")

    # (3) tree-filtered stats (optional)
    if tree_filter is not None:
        print(f"[3] PAT vs Baseline: kernel latency reduction (%) within tree='{tree_filter}'")
        print(f"{'Baseline':<12}  {'Mean':>10}  {'Min':>10}  {'Max':>10}  {'N':>6}")
        print("-" * 56)
        stats_out["reduction_tree"] = {}
        for b in baseline_keys:
            xs = red_tree.get(b, [])
            if not xs:
                print(f"{b:<12}  {'NA':>10}  {'NA':>10}  {'NA':>10}  {0:>6}")
                continue
            mean_v, min_v, max_v = _safe_stats(xs)
            print(
                f"{b:<12}  {_format_pct(mean_v, digits_pct):>10}  "
                f"{_format_pct(min_v, digits_pct):>10}  {_format_pct(max_v, digits_pct):>10}  {len(xs):>6}"
            )
            stats_out["reduction_tree"][b] = {"mean": mean_v, "min": min_v, "max": max_v, "n": len(xs)}

        print("")

    return stats_out


def summarize_kernel_perf_for_tree(
    log_file: str,
    tree: str,
    pat_key: str = "pat",
    baseline_keys: Optional[List[str]] = None,
    digits_pct: int = 2,
):
    """
    Convenience wrapper for requirement (3):
      Under a specified tree config, summarize PAT reduction vs each baseline across all head configs.
    """
    return summarize_kernel_perf(
        log_file=log_file,
        pat_key=pat_key,
        baseline_keys=baseline_keys,
        tree_filter=tree,
        digits_pct=digits_pct,
    )


def create_performance_plot(
    filename,
    operator_order=None,
    selected_indices=None,
    list_trees_only=False,
    labels=None,
):
    """
    read JSON data and generate performance plot.

    Args:
        filename (str): input JSON filename.
        operator_order (list, optional): specify the order of operators (keys in JSON) to show in the plot.
        selected_indices (list, optional): list of integers. Corresponding to the order of Trees.
        list_trees_only (bool): only print the list for viewing indices.
    """

    try:
        with open(filename, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: file '{filename}' is not a valid JSON file.")
        return

    # transform raw data into DataFrame
    processed_rows = []
    for entry in raw_data:
        row = {}
        row["tree"] = entry.get("tree", "Unknown")

        if "nheads_q" in entry and "nheads_kv" in entry:
            row["hq/hkv"] = f"{entry['nheads_q']}/{entry['nheads_kv']}"
        else:
            row["hq/hkv"] = "Unknown"

        latencies = entry.get("latencies", {})
        for op_name, op_val in latencies.items():
            row[op_name] = op_val

        processed_rows.append(row)

    df = pd.DataFrame(processed_rows)

    if "tree" not in df.columns:
        print("Error: 'tree' column not found in the data.")
        return

    # filter out rows with missing 'tree' values
    df.dropna(subset=["tree"], inplace=True)

    # --- Key change: directly obtain the Tree list while preserving the original order ---
    # Pandas unique() preserves unique values in their order of appearance
    all_unique_trees = df["tree"].unique().tolist()

    if not all_unique_trees:
        print("Error: no valid Tree data found.")
        return

    # --- Mode 1: discovery mode ---
    if list_trees_only:
        print(f"--- Original order in JSON file (Total: {len(all_unique_trees)}) ---")
        for i, tree in enumerate(all_unique_trees):
            print(f"  Index {i}: {tree}")
        print("-" * 40)
        print("Use the [index numbers] above in indices to choose data to plot.")
        return

    # --- Mode 2: plotting mode ---

    # 1. Determine which Trees to plot based on selected_indices
    x_ticks, trees_to_plot = [], []  # Stores x-axis labels

    if selected_indices is None:
        # If not specified, plot everything while keeping file order
        trees_to_plot = all_unique_trees
        print(
            f"Info: no indices specified, plotting all {len(trees_to_plot)} Trees in file order."
        )

        # Generate labels
        for tree in trees_to_plot:
            t = tree.split("_")
            if len(t) >= 2:
                x_ticks.append(f"B=[{t[0]}]\nL=[{t[1]}]")
            else:
                x_ticks.append(tree)
    else:
        # Extract according to the user-provided indices while maintaining their order
        invalid_indices = []
        for index in selected_indices:
            if 0 <= index < len(all_unique_trees):
                tree_name = all_unique_trees[index]
                trees_to_plot.append(tree_name)

                # Format x-axis labels
                t = tree_name.split("_")
                if len(t) >= 2:
                    x_ticks.append(f"B=[{t[0]}]\nL=[{t[1]}]")
                else:
                    x_ticks.append(tree_name)
            else:
                invalid_indices.append(index)

        if invalid_indices:
            print(
                f"Warning: indices {invalid_indices} are out of range and were ignored."
            )

        print(f"Info: found {len(trees_to_plot)} Trees for plotting.")

    if not trees_to_plot:
        print("Error: no valid Trees available for plotting.")
        return

    # 2. Create a mapping: map Tree names to 0, 1, 2... (for x-axis placement)
    # Here, i corresponds to the order in trees_to_plot (i.e., the user's selected order)
    tree_to_num = {tree: i for i, tree in enumerate(trees_to_plot)}

    # 3. Determine operator columns
    default_operators = ["fa", "vllm-fa", "flashinfer", "ra", "ra++", "FastTree", "pat"]
    if operator_order is None:
        operators = default_operators
    else:
        operators = [op for op in operator_order if op in df.columns]

    if not operators:
        print("Error: no matching operator data columns found.")
        return

    # 4. Determine group (hq/hkv) order, also preserving the file order
    h_q_hkv_groups = df["hq/hkv"].unique().tolist()

    # Begin plotting
    fig, axes = plt.subplots(
        len(h_q_hkv_groups),
        1,
        figsize=(12, 1.25 * len(h_q_hkv_groups)),
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for i, group in enumerate(h_q_hkv_groups):
        ax = axes[i]
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.5)

        # Filter the data for the current group
        # Note: filter by group first, then by whether it is in selected_indices
        group_df = df[(df["hq/hkv"] == group) & (df["tree"].isin(trees_to_plot))].copy()

        plot_df = pd.DataFrame()
        if not group_df.empty:
            # Normalize the data (1 / latency)
            subset = group_df.set_index("tree")[operators].reindex(columns=operators)
            subset = subset.apply(pd.to_numeric, errors="coerce")  # Convert to numeric

            reciprocal_df = 1 / subset
            max_reciprocal_per_row = reciprocal_df.max(axis=1)
            normalized_df = reciprocal_df.div(max_reciprocal_per_row, axis=0)

            # Map to plotting indices (x-axis positions)
            normalized_df["tree"] = normalized_df.index
            normalized_df["tree_num"] = normalized_df["tree"].map(tree_to_num)

            # Must be sorted by tree_num to keep bar order aligned with x-axis labels
            plot_df = normalized_df.set_index("tree_num").sort_index()

        if not plot_df.empty:
            plot_df[operators].plot(
                kind="bar",
                ax=ax,
                width=0.7,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.1,
            )

            if ax.get_legend() is not None:
                ax.get_legend().remove()

            # --- Draw an 'X' for missing data ---
            bar_group_width = 0.8
            bar_width = bar_group_width / len(operators)

            for tree_idx, row in plot_df.iterrows():
                for op_idx, op_name in enumerate(operators):
                    if pd.isna(row[op_name]):
                        # tree_idx represents 0, 1, 2... corresponding to integer x-axis positions
                        x_pos = (
                            tree_idx
                            - (bar_group_width / 2)
                            + (op_idx * bar_width)
                            + (bar_width / 2)
                        )
                        ax.text(
                            x_pos,
                            0.04,
                            "X",
                            ha="center",
                            va="center",
                            color="red",
                            fontsize=5,
                            fontweight="bold",
                        )
        else:
            ax.text(
                0.5,
                0.5,
                "No Data for Selected Trees",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )

        # Set the Y-axis
        ax.set_yticks([0.5, 1])
        ax.set_yticklabels(["50%", "100%"])
        # ax.set_ylabel(f'Hq/Mkv={group}', fontsize=10, fontweight='bold')

        # Configure the X-axis (only on the last subplot)
        if i == len(h_q_hkv_groups) - 1:
            # Set tick positions to 0 through N-1
            ax.set_xticks(range(len(trees_to_plot)))
            # Set tick labels
            ax.set_xticklabels(x_ticks, fontsize=9.5, rotation=40, ha="right")

            # Add purple circle indices
            for j, tree in enumerate(trees_to_plot):
                # j is the x-axis position, and tree_to_num[tree] is also j (because we generated it sequentially)
                # The displayed number is j + 1
                ax.text(
                    j,
                    -0.08,
                    str(j + 1),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="purple",
                    bbox=dict(
                        facecolor="white", edgecolor="purple", boxstyle="circle,pad=0.2"
                    ),
                    transform=ax.get_xaxis_transform(),
                )  # Ensures the position is relative to the X-axis
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", length=0)

        ax.tick_params(axis="x", which="both", length=0)

    # Global Y-axis label
    # fig.text(0.02, 0.5, 'Normalized Kernel Performance (Higher is Better)', va='center', rotation='vertical',
    #          fontsize=14)
    fig.supylabel("normalized kernel performance (higher is better)", fontsize=14)
    plt.xlabel("")

    # --- Legend ---
    handles, _labels = (
        axes[0].get_legend_handles_labels() if axes.size > 0 else (None, None)
    )
    if handles:
        final_labels = labels if labels and len(labels) == len(operators) else operators
        fig.legend(
            handles,
            final_labels,
            loc="upper center",
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(operators),
        )

    # Save the figure
    output_dir = "fig"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = f"{output_dir}/eval_kernel_overall.pdf"

    plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.31, hspace=0.05)
    plt.savefig(output_filename)
    plt.show()
    print(f"\nImage saved: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="plot kernel performance from JSON log file"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="results/A100_kernel_perf.json",
        help="log file in JSON format",
    )
    args = parser.parse_args()
    json_file = args.log_file

    # create_performance_plot(filename=json_file, list_trees_only=True)
    create_performance_plot(
        filename=json_file,
        operator_order=["pat", "FastTree", "ra", "ra++", "flashinfer", "vllm-fa"],
        selected_indices=list(range(20)),  # None means all
        labels=[
            "PAT",
            "FastTree",
            "RelayAttn",
            "RelayAttn++",
            "FlashInfer",
            "FlashAttn",
        ],
    )
    
    # Summarize kernel performance
    baseline_keys = ["flashinfer", "vllm-fa", "ra", "ra++", "FastTree"]
    # summarize_kernel_perf_for_tree(json_file, baseline_keys=baseline_keys, tree="256_1024", pat_key="pat")
    # summarize_kernel_perf_for_tree(json_file, baseline_keys=baseline_keys, tree="256_4096", pat_key="pat")
    summarize_kernel_perf(json_file, baseline_keys=baseline_keys, pat_key="pat")
