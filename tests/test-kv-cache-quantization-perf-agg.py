#!/usr/bin/env python3
"""
Aggregate multiple KV cache quantization performance CSV files.

Reads CSVs produced by test-kv-cache-quantization-perf.sh, groups rows by
their grouping key (config, coopmat_mode, cache_k, cache_v), and computes
aggregated mean/stdev across runs. Outputs a combined CSV and a rendered
text table, appending the list of source files.

Usage:
    python tests/test-kv-cache-quantization-perf-agg.py -o aggregated.csv file1.csv file2.csv ...
    python tests/test-kv-cache-quantization-perf-agg.py -o aggregated.csv kv-perf_*.csv
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict


GROUP_KEYS = ("config", "coopmat_mode", "cache_k", "cache_v")
PASSTHROUGH = ("gpu_device", "model", "mixed", "kv_size_mib", "compression_vs_f16",
               "prompt_len", "gen_len")
AGG_COLS = ("pp_avg", "pp_stdev", "tg_avg", "tg_stdev")
RATIO_COLS = ("pp_vs_f16_x", "tg_vs_f16_x", "pp_speedup_vs_scalar", "tg_speedup_vs_scalar")

OUTPUT_HEADER = [
    "gpu_device", "model", "config", "coopmat_mode", "cache_k", "cache_v",
    "mixed", "kv_size_mib", "compression_vs_f16", "prompt_len", "gen_len",
    "n_runs", "pp_avg", "pp_stdev", "tg_avg", "tg_stdev",
    "pp_vs_f16_x", "tg_vs_f16_x", "pp_speedup_vs_scalar", "tg_speedup_vs_scalar",
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
        "pp_means": [], "pp_stdevs": [],
        "tg_means": [], "tg_stdevs": [],
        "passthrough": None,
        "reps_total": 0,
    })

    for fpath in input_files:
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = tuple(row.get(k, "").strip().strip('"') for k in GROUP_KEYS)

                pp_avg = safe_float(row.get("pp_avg"))
                pp_sd = safe_float(row.get("pp_stdev"))
                tg_avg = safe_float(row.get("tg_avg"))
                tg_sd = safe_float(row.get("tg_stdev"))

                if pp_avg is None or tg_avg is None:
                    continue

                g = groups[key]
                g["pp_means"].append(pp_avg)
                g["pp_stdevs"].append(pp_sd)
                g["tg_means"].append(tg_avg)
                g["tg_stdevs"].append(tg_sd)

                reps = safe_float(row.get("reps"))
                g["reps_total"] += int(reps) if reps else 1

                if g["passthrough"] is None:
                    g["passthrough"] = {k: row.get(k, "").strip().strip('"') for k in PASSTHROUGH}

    return groups


def compute_ratios(groups):
    """Recompute vs-f16 and vs-scalar ratios from aggregated values."""
    f16_lookup = {}
    for key, g in groups.items():
        config, cm_mode, ck, cv = key
        if ck == "f16" and cv == "f16":
            f16_lookup[(config, cm_mode)] = g

    results = []
    for key in groups:
        config, cm_mode, ck, cv = key
        g = groups[key]

        pp_mean, pp_sd = combined_mean_stdev(g["pp_means"], g["pp_stdevs"])
        tg_mean, tg_sd = combined_mean_stdev(g["tg_means"], g["tg_stdevs"])
        n_runs = len(g["pp_means"])

        pp_vs_f16 = ""
        tg_vs_f16 = ""
        f16 = f16_lookup.get((config, cm_mode))
        if f16 is not None:
            f16_pp, _ = combined_mean_stdev(f16["pp_means"], f16["pp_stdevs"])
            f16_tg, _ = combined_mean_stdev(f16["tg_means"], f16["tg_stdevs"])
            if f16_pp and f16_pp > 0:
                pp_vs_f16 = f"{pp_mean / f16_pp:.2f}"
            if f16_tg and f16_tg > 0:
                tg_vs_f16 = f"{tg_mean / f16_tg:.2f}"

        scalar_key = (config, "scalar", ck, cv)
        pp_vs_scalar = ""
        tg_vs_scalar = ""
        if cm_mode != "scalar" and scalar_key in groups:
            sg = groups[scalar_key]
            s_pp, _ = combined_mean_stdev(sg["pp_means"], sg["pp_stdevs"])
            s_tg, _ = combined_mean_stdev(sg["tg_means"], sg["tg_stdevs"])
            if s_pp and s_pp > 0:
                pp_vs_scalar = f"{pp_mean / s_pp:.2f}"
            if s_tg and s_tg > 0:
                tg_vs_scalar = f"{tg_mean / s_tg:.2f}"

        pt = g["passthrough"] or {}
        results.append({
            "gpu_device": pt.get("gpu_device", ""),
            "model": pt.get("model", ""),
            "config": config,
            "coopmat_mode": cm_mode,
            "cache_k": ck,
            "cache_v": cv,
            "mixed": pt.get("mixed", ""),
            "kv_size_mib": pt.get("kv_size_mib", ""),
            "compression_vs_f16": pt.get("compression_vs_f16", ""),
            "prompt_len": pt.get("prompt_len", ""),
            "gen_len": pt.get("gen_len", ""),
            "n_runs": str(n_runs),
            "pp_avg": f"{pp_mean:.2f}" if pp_mean is not None else "",
            "pp_stdev": f"{pp_sd:.2f}" if pp_sd is not None else "",
            "tg_avg": f"{tg_mean:.2f}" if tg_mean is not None else "",
            "tg_stdev": f"{tg_sd:.2f}" if tg_sd is not None else "",
            "pp_vs_f16_x": pp_vs_f16,
            "tg_vs_f16_x": tg_vs_f16,
            "pp_speedup_vs_scalar": pp_vs_scalar,
            "tg_speedup_vs_scalar": tg_vs_scalar,
        })

    return results


def render_table(rows):
    """Render rows as a fixed-width text table."""
    if not rows:
        return ""

    display_cols = [
        ("config", "Config", 8),
        ("coopmat_mode", "CM Mode", 10),
        ("cache_k", "K", 10),
        ("cache_v", "V", 10),
        ("compression_vs_f16", "Compr", 7),
        ("n_runs", "Runs", 5),
        ("pp_avg", "pp avg", 10),
        ("pp_stdev", "pp sd", 8),
        ("tg_avg", "tg avg", 10),
        ("tg_stdev", "tg sd", 8),
        ("pp_vs_f16_x", "pp/f16", 7),
        ("tg_vs_f16_x", "tg/f16", 7),
        ("pp_speedup_vs_scalar", "pp/scl", 7),
        ("tg_speedup_vs_scalar", "tg/scl", 7),
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
        description="Aggregate KV cache quantization performance CSVs")
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
        f.write(" KV Cache Quantization Performance — Aggregated Results\n")
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
