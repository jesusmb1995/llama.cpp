#!/usr/bin/env bash
#
# Needle-In-A-Haystack (NIAH) benchmark for KV cache quantization.
#
# Runs llama-passkey at multiple haystack sizes and needle positions,
# then reports per-cell accuracy with mean +/- stdev.
#
# Usage (direct):
#   ./tests/niah-bench.sh -m model.gguf [options]
#
# Usage (sourced):
#   source tests/niah-bench.sh
#   niah_bench -m model.gguf -ctk tbq3_0 -ctv tbq3_0 -ngl 99
#   echo "accuracy: $niah_global_pct ± $niah_global_stdev%"
#
# All unknown flags are forwarded to llama-passkey (e.g. -ngl, --threads, etc.)
#
# After niah_bench returns, the following variables are set:
#   niah_global_pct    - overall accuracy %       (e.g. "100.0")
#   niah_global_stdev  - SE of global accuracy %  (e.g. "0.0000"), use as global_pct ± global_stdev
#   niah_global_pass   - total passing trials
#   niah_global_total  - total trials
#   niah_cell_mean     - per-cell mean accuracy %  (e.g. "100.0000")
#   niah_cell_stdev    - per-cell stdev %  (e.g. "0.0000"), measures uniformity across cells
#   niah_n_cells       - number of (junk_size, depth) cells

niah_usage() {
    cat <<'EOF'
Needle-In-A-Haystack benchmark for KV cache quantization

Required:
  -m, --model PATH          Path to GGUF model

Options:
  -ctk  TYPE                KV cache type for K          (default: f16)
  -ctv  TYPE                KV cache type for V          (default: f16)
  --junk-sizes "N ..."      Space-separated junk sizes   (default: "50 100 250 500")
  --depths     "N ..."      Depth positions as % 0-100   (default: "0 25 50 75 100")
  --repeats    N            Trials per cell               (default: 5)
  --n-ctx      N            Force context size            (default: auto)
  --seed-base  N            Starting seed                 (default: 42)
  --passkey-bin PATH        Path to llama-passkey binary  (default: build/bin/llama-passkey)
  -q, --quiet               Suppress per-trial output
  -h, --help                Show this help

All other flags are forwarded to llama-passkey (e.g. -ngl 99, --threads 8).

Example:
  ./tests/niah-bench.sh -m model.gguf -ctk tbq3_0 -ctv tbq3_0 --repeats 10 -ngl 99
EOF
}

_niah_run_single() {
    local passkey_bin=$1 model=$2 ctk=$3 ctv=$4 junk=$5 pos=$6 seed=$7 n_ctx=$8
    shift 8
    local extra_args=("$@")

    local ctx=${n_ctx:-$(( (junk + 10) * 30 ))}
    local output_file
    local expected actual
    output_file=$(mktemp)

    {
        "$passkey_bin" \
            -m "$model" \
            -ctk "$ctk" -ctv "$ctv" \
            --junk "$junk" --pos "$pos" \
            --seed "$seed" \
            -c "$ctx" \
            "${extra_args[@]}" 2>&1 \
            | tee "$output_file" \
            | grep -E -i 'passkey =|pass key is|n_len|decoded|shifting kv cache' || true
    } >&2 || true

    # "main: passkey = 39384, inserted at position ..."
    expected=$(grep -oP 'passkey\s*=\s*\K[0-9]+' "$output_file" | head -1)
    # " What is the pass key? The pass key is 39384."
    actual=$(grep -oP 'The\s+pass\s+key\s+is\s*\K[0-9]+' "$output_file" | tail -1)
    if [[ -z "$actual" ]]; then
        actual=$(grep -oP 'pass\s+key\s+is\s*\K[0-9]+' "$output_file" | tail -1)
    fi

    if [[ -z "$expected" || -z "$actual" ]]; then
        echo "FAIL:parse"
        rm -f "$output_file"
        return
    fi

    rm -f "$output_file"

    if [[ "$expected" == "$actual" ]]; then
        echo "PASS"
    else
        echo "FAIL:${expected}!=${actual}"
    fi
}

niah_bench() {
    # ── defaults ──────────────────────────────────────────────────────────────
    local passkey_bin="build/bin/llama-passkey"
    local model=""
    local ctk="f16"
    local ctv="f16"
    local junk_sizes="50 100 250 500"
    local depths="0 25 50 75 100"
    local repeats=5
    local n_ctx=""
    local seed_base=42
    local extra_args=()
    local quiet=0

    # ── argument parsing ──────────────────────────────────────────────────────
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)       niah_usage; return 0 ;;
            -m|--model)      model="$2";       shift 2 ;;
            -ctk)            ctk="$2";         shift 2 ;;
            -ctv)            ctv="$2";         shift 2 ;;
            --junk-sizes)    junk_sizes="$2";  shift 2 ;;
            --depths)        depths="$2";      shift 2 ;;
            --repeats)       repeats="$2";     shift 2 ;;
            --n-ctx)         n_ctx="$2";       shift 2 ;;
            --seed-base)     seed_base="$2";   shift 2 ;;
            --passkey-bin)   passkey_bin="$2"; shift 2 ;;
            -q|--quiet)      quiet=1;          shift ;;
            *)               extra_args+=("$1"); shift ;;
        esac
    done

    if [[ -z "$model" ]]; then
        echo "ERROR: --model is required" >&2
        niah_usage >&2
        return 1
    fi

    if [[ ! -x "$passkey_bin" ]]; then
        echo "ERROR: passkey binary not found at '$passkey_bin'" >&2
        echo "       Build with: cmake --build build --target llama-passkey" >&2
        return 1
    fi

    # ── banner ────────────────────────────────────────────────────────────────
    echo "=========================================="
    echo " Needle-In-A-Haystack Benchmark"
    echo "=========================================="
    echo "  Model:    $(basename "$model")"
    echo "  K type:   $ctk"
    echo "  V type:   $ctv"
    echo "  Junks:    $junk_sizes"
    echo "  Depths:   $depths"
    echo "  Repeats:  $repeats"
    echo "  Seed:     $seed_base"
    [[ ${#extra_args[@]} -gt 0 ]] && echo "  Extra:    ${extra_args[*]}"
    echo "=========================================="
    echo ""

    # ── main benchmark loop ───────────────────────────────────────────────────
    declare -A cell_pass cell_total
    local global_pass=0
    local global_total=0
    local n_junk n_depth total_trials trial_idx overall_pct
    n_junk=$(wc -w <<< "$junk_sizes")
    n_depth=$(wc -w <<< "$depths")
    total_trials=$(( n_junk * n_depth * repeats ))
    trial_idx=0

    local junk depth_pct pos cell_key trial seed result tag
    for junk in $junk_sizes; do
        for depth_pct in $depths; do
            pos=$(( junk * depth_pct / 100 ))
            if (( pos >= junk )); then
                pos=$(( junk - 1 ))
            fi

            cell_key="${junk}_${depth_pct}"
            cell_pass[$cell_key]=0
            cell_total[$cell_key]=0

            for (( trial=0; trial < repeats; trial++ )); do
                seed=$(( seed_base + trial ))
                result=$(_niah_run_single "$passkey_bin" "$model" "$ctk" "$ctv" \
                         "$junk" "$pos" "$seed" "$n_ctx" "${extra_args[@]}")
                result="${result//$'\r'/}"
                result=$(printf '%s\n' "$result" | awk 'NF{line=$0} END{print line}')

                if [[ "$result" == "PASS" ]]; then
                    cell_pass[$cell_key]=$(( ${cell_pass[$cell_key]} + 1 ))
                    global_pass=$(( global_pass + 1 ))
                    tag="OK"
                else
                    tag="MISS"
                fi
                cell_total[$cell_key]=$(( ${cell_total[$cell_key]} + 1 ))
                global_total=$(( global_total + 1 ))
                trial_idx=$(( trial_idx + 1 ))
                if (( total_trials > 0 )); then
                    overall_pct=$(( trial_idx * 100 / total_trials ))
                else
                    overall_pct=0
                fi

                if [[ $quiet -eq 0 ]]; then
                    printf "  [%3d%% %3d/%3d] [junk=%3d depth=%3d%% trial=%d] %s  (%s)\n" \
                        "$overall_pct" "$trial_idx" "$total_trials" \
                        "$junk" "$depth_pct" "$((trial+1))" "$tag" "$result"
                fi
            done
        done
    done

    # ── results table ─────────────────────────────────────────────────────────
    echo ""
    echo "=========================================="
    echo " Results: K=$ctk  V=$ctv"
    echo "=========================================="
    echo ""

    printf "%-12s" "Junk\\Depth"
    for depth_pct in $depths; do
        printf "%10s" "${depth_pct}%"
    done
    printf "%12s\n" "Row Avg"

    local total_cols=$(( $(echo "$depths" | wc -w) + 2 ))
    printf '%*s\n' $(( total_cols * 10 + 12 )) '' | tr ' ' '-'

    local p t pct row_pass row_total row_pct
    for junk in $junk_sizes; do
        printf "%-12s" "junk=$junk"
        row_pass=0
        row_total=0

        for depth_pct in $depths; do
            cell_key="${junk}_${depth_pct}"
            p=${cell_pass[$cell_key]}
            t=${cell_total[$cell_key]}
            pct=$(awk "BEGIN { printf \"%.0f\", ($p / $t) * 100 }")
            printf "%9s%%" "$pct"
            row_pass=$(( row_pass + p ))
            row_total=$(( row_total + t ))
        done

        row_pct=$(awk "BEGIN { printf \"%.0f\", ($row_pass / $row_total) * 100 }")
        printf "%11s%%\n" "$row_pct"
    done

    echo ""

    # ── global statistics with stdev ──────────────────────────────────────────
    local cell_rates="" n_cells=0 rate
    for junk in $junk_sizes; do
        for depth_pct in $depths; do
            cell_key="${junk}_${depth_pct}"
            p=${cell_pass[$cell_key]}
            t=${cell_total[$cell_key]}
            rate=$(awk "BEGIN { printf \"%.6f\", $p / $t }")
            cell_rates="$cell_rates $rate"
            n_cells=$(( n_cells + 1 ))
        done
    done

    local stats
    stats=$(echo "$cell_rates" | awk '{
        n = NF
        sum = 0
        for (i = 1; i <= n; i++) sum += $i
        mean = sum / n
        sumsq = 0
        for (i = 1; i <= n; i++) sumsq += ($i - mean)^2
        sd = (n > 1) ? sqrt(sumsq / (n - 1)) : 0
        printf "%.4f %.4f", mean * 100, sd * 100
    }')

    # ── export results to caller ──────────────────────────────────────────────
    niah_global_pass=$global_pass
    niah_global_total=$global_total
    niah_global_pct=$(awk "BEGIN { printf \"%.1f\", ($global_pass / $global_total) * 100 }")
    # SE of a Bernoulli proportion: sqrt(p*(1-p)/N)
    niah_global_stdev=$(awk "BEGIN {
        p = $global_pass / $global_total
        printf \"%.4f\", sqrt(p * (1 - p) / $global_total) * 100
    }")
    niah_cell_mean=${stats% *}
    niah_cell_stdev=${stats#* }
    niah_n_cells=$n_cells

    echo "=========================================="
    echo " Summary"
    echo "=========================================="
    echo "  Global accuracy:     ${niah_global_pct} ± ${niah_global_stdev}%  (${niah_global_pass}/${niah_global_total} trials)"
    echo "  Per-cell uniformity: ${niah_cell_mean} ± ${niah_cell_stdev}%  (mean ± stdev across ${niah_n_cells} cells)"
    echo "=========================================="
}

# ── auto-run when executed directly (not sourced) ─────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    niah_bench "$@"
fi
