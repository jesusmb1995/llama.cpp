#!/usr/bin/env python3
"""
Aggregate multiple KV cache quantization perplexity CSV files.

Reads CSVs produced by test-kv-cache-quantization-perp.sh, groups rows by
their grouping key (config, cache_k, cache_v, norm_correction), and computes
aggregated mean/stdev across runs. Outputs a combined CSV and a rendered
text table, appending the list of source files.

Usage:
    python tests/test-kv-cache-quantization-perp-agg.py -o aggregated.csv file1.csv file2.csv ...
    python tests/test-kv-cache-quantization-perp-agg.py -o aggregated.csv kv-perp_*.csv
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict


GROUP_KEYS = ("config", "cache_k", "cache_v", "norm_correction")
PASSTHROUGH = ("gpu_device", "model", "mixed", "bpw_avg", "n_chunks", "n_ctx_sweep")

OUTPUT_HEADER = [
    "gpu_device", "model", "config", "cache_k", "cache_v", "mixed",
    "norm_correction", "bpw_avg", "n_chunks", "n_ctx_sweep",
    "n_runs", "ppl_mean", "ppl_stdev", "time_mean_s", "time_stdev_s",
    "ppl_vs_f16_pct", "ppl_vs_f16_pct_stdev",
    "time_vs_f16_x", "time_vs_f16_x_stdev",
]


def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def combined_mean_stdev(means, stdevs):
    """Combine per-run mean±stdev into an aggregate mean and stdev.

    Uses the law of total variance:
        combined_var = mean(individual_variances) + var(individual_means)
    """
    n = len(means)
    if n == 0:
        return None, None
    grand_mean = sum(means) / n
    if n == 1:
        return grand_mean, stdevs[0] if stdevs[0] is not None else 0.0

    mean_of_vars = sum((s ** 2 if s is not None else 0.0) for s in stdevs) / n
    var_of_means = sum((m - grand_mean) ** 2 for m in means) / (n - 1)
    combined_sd = math.sqrt(mean_of_vars + var_of_means)
    return grand_mean, combined_sd


def aggregate(input_files):
    groups = defaultdict(lambda: {
        "ppl_means": [], "ppl_stdevs": [],
        "time_means": [], "time_stdevs": [],
        "passthrough": None,
    })

    for fpath in input_files:
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = tuple(row.get(k, "").strip().strip('"') for k in GROUP_KEYS)

                ppl_mean = safe_float(row.get("ppl_mean"))
                ppl_sd = safe_float(row.get("ppl_stdev"))
                time_mean = safe_float(row.get("time_mean_s"))
                time_sd = safe_float(row.get("time_stdev_s"))

                if ppl_mean is None:
                    continue

                g = groups[key]
                g["ppl_means"].append(ppl_mean)
                g["ppl_stdevs"].append(ppl_sd)
                if time_mean is not None:
                    g["time_means"].append(time_mean)
                    g["time_stdevs"].append(time_sd)

                if g["passthrough"] is None:
                    g["passthrough"] = {k: row.get(k, "").strip().strip('"') for k in PASSTHROUGH}

    return groups


def compute_ratios(groups):
    """Recompute vs-f16 ratios from aggregated values."""
    f16_lookup = {}
    for key, g in groups.items():
        config, ck, cv, nc = key
        if ck == "f16" and cv == "f16":
            f16_lookup[(config, nc)] = g

    results = []
    for key in groups:
        config, ck, cv, nc = key
        g = groups[key]

        ppl_mean, ppl_sd = combined_mean_stdev(g["ppl_means"], g["ppl_stdevs"])
        n_runs = len(g["ppl_means"])

        time_mean, time_sd = None, None
        if g["time_means"]:
            time_mean, time_sd = combined_mean_stdev(g["time_means"], g["time_stdevs"])

        ppl_vs_f16 = ""
        ppl_vs_f16_sd = ""
        time_vs_f16 = ""
        time_vs_f16_sd = ""

        f16 = f16_lookup.get((config, nc))
        if f16 is None:
            f16 = f16_lookup.get((config, "off"))

        if f16 is not None:
            f16_ppl, _ = combined_mean_stdev(f16["ppl_means"], f16["ppl_stdevs"])
            if f16_ppl and f16_ppl > 0:
                pct = ((ppl_mean - f16_ppl) / f16_ppl) * 100
                ppl_vs_f16 = f"{pct:.2f}"
                if ppl_sd is not None:
                    ppl_vs_f16_sd = f"{(ppl_sd / f16_ppl) * 100:.2f}"

            if time_mean is not None and f16["time_means"]:
                f16_time, f16_time_sd = combined_mean_stdev(
                    f16["time_means"], f16["time_stdevs"])
                if f16_time and f16_time > 0:
                    ratio = time_mean / f16_time
                    time_vs_f16 = f"{ratio:.2f}"
                    if time_sd is not None and f16_time_sd is not None and f16_time > 0:
                        a = (time_sd / f16_time) ** 2
                        b = (time_mean * f16_time_sd / (f16_time ** 2)) ** 2
                        time_vs_f16_sd = f"{math.sqrt(a + b):.2f}"

        pt = g["passthrough"] or {}
        results.append({
            "gpu_device": pt.get("gpu_device", ""),
            "model": pt.get("model", ""),
            "config": config,
            "cache_k": ck,
            "cache_v": cv,
            "mixed": pt.get("mixed", ""),
            "norm_correction": nc,
            "bpw_avg": pt.get("bpw_avg", ""),
            "n_chunks": pt.get("n_chunks", ""),
            "n_ctx_sweep": pt.get("n_ctx_sweep", ""),
            "n_runs": str(n_runs),
            "ppl_mean": f"{ppl_mean:.4f}" if ppl_mean is not None else "",
            "ppl_stdev": f"{ppl_sd:.4f}" if ppl_sd is not None else "",
            "time_mean_s": f"{time_mean:.2f}" if time_mean is not None else "",
            "time_stdev_s": f"{time_sd:.2f}" if time_sd is not None else "",
            "ppl_vs_f16_pct": ppl_vs_f16,
            "ppl_vs_f16_pct_stdev": ppl_vs_f16_sd,
            "time_vs_f16_x": time_vs_f16,
            "time_vs_f16_x_stdev": time_vs_f16_sd,
        })

    return results


def render_table(rows):
    """Render rows as a fixed-width text table."""
    if not rows:
        return ""

    display_cols = [
        ("config", "Config", 8),
        ("cache_k", "K", 10),
        ("cache_v", "V", 10),
        ("norm_correction", "NC", 4),
        ("bpw_avg", "BPW", 6),
        ("n_runs", "Runs", 5),
        ("ppl_mean", "PPL mean", 10),
        ("ppl_stdev", "PPL sd", 8),
        ("ppl_vs_f16_pct", "vs f16%", 8),
        ("ppl_vs_f16_pct_stdev", "±%", 6),
        ("time_mean_s", "Time(s)", 8),
        ("time_stdev_s", "Time sd", 8),
        ("time_vs_f16_x", "t/f16x", 7),
    ]

    lines = []
    header = "  ".join(f"{title:>{w}}" for _, title, w in display_cols)
    lines.append(header)
    lines.append("  ".join("-" * w for _, _, w in display_cols))

    for row in rows:
        line = "  ".join(f"{row.get(k, ''):>{w}}" for k, _, w in display_cols)
        lines.append(line)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate KV cache quantization perplexity CSVs")
    parser.add_argument("inputs", nargs="+", help="Input CSV files")
    parser.add_argument("-o", "--output", required=True,
                        help="Output CSV file path")
    args = parser.parse_args()

    missing = [f for f in args.inputs if not os.path.isfile(f)]
    if missing:
        print(f"Error: files not found: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    groups = aggregate(args.inputs)
    results = compute_ratios(groups)

    out_csv = args.output
    out_txt = os.path.splitext(out_csv)[0] + ".txt"

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER,
                                quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(results)

    table = render_table(results)
    with open(out_txt, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(" KV Cache Quantization Perplexity — Aggregated Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(table + "\n\n")
        f.write("=" * 80 + "\n")
        f.write(f" Aggregated from {len(args.inputs)} file(s):\n")
        for fpath in args.inputs:
            f.write(f"   - {os.path.abspath(fpath)}\n")
        f.write("=" * 80 + "\n")

    print(f"CSV: {os.path.abspath(out_csv)} ({len(results)} rows)")
    print(f"TXT: {os.path.abspath(out_txt)}")
    print()
    print(table)
    print()
    print(f"Aggregated from {len(args.inputs)} file(s):")
    for fpath in args.inputs:
        print(f"  - {fpath}")


if __name__ == "__main__":
    main()
