#!/usr/bin/env bash
#
# Benchmark KV cache quantization throughput (prefill pp & decode tg t/s).
# Compares cache types with coopmat2 enabled vs disabled to measure the
# coopmat2 FA speedup for TBQ/PQ quantizations.
#
# Usage:
#   tests/test-kv-cache-quantization-perf.sh [build_dir]
#
# Environment variables:
#   MODEL_DIR    — directory for model files (default: models)
#   MODEL_NAME   — GGUF filename
#   MODEL_URL    — download URL
#   PROMPT_LEN   — prompt length for prefill benchmark (default: 512)
#   GEN_LEN      — number of tokens to generate for decode benchmark (default: 128)
#   REPS         — number of repetitions per config (default: 1)
#   SKIP_COOPMAT_COMPARE — set to 1 to skip the coopmat-disabled runs
#
set -euo pipefail

BUILD_DIR="${1:-build}"
BENCH="$BUILD_DIR/bin/llama-bench"

MODEL_DIR="${MODEL_DIR:-models}"
MODEL_NAME="${MODEL_NAME:-Mistral-7B-Instruct-v0.3-Q4_K_S.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_S.gguf}"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

PROMPT_LEN="${PROMPT_LEN:-512}"
GEN_LEN="${GEN_LEN:-128}"
REPS="${REPS:-1}"
SKIP_COOPMAT_COMPARE="${SKIP_COOPMAT_COMPARE:-0}"

# --- Validate ---

if [ ! -f "$BENCH" ]; then
    echo "Error: $BENCH not found. Build the project first."
    echo "  cmake -B $BUILD_DIR && cmake --build $BUILD_DIR -j\$(nproc)"
    exit 1
fi

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ] || [ "$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)" -lt 1000000 ]; then
    [ -f "$MODEL_PATH" ] && rm -f "$MODEL_PATH"
    echo "Downloading model: $MODEL_NAME ..."
    if command -v curl &> /dev/null; then
        curl -L --fail -C - -o "$MODEL_PATH" "$MODEL_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$MODEL_PATH" "$MODEL_URL"
    else
        echo "Error: neither curl nor wget found"
        exit 1
    fi
    FILE_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -lt 1000000 ]; then
        echo "Error: downloaded file is only $FILE_SIZE bytes."
        rm -f "$MODEL_PATH"
        exit 1
    fi
    echo "Model downloaded to $MODEL_PATH ($((FILE_SIZE / 1048576)) MB)"
else
    echo "Model already exists: $MODEL_PATH"
fi

# --- Config ---

SAME_TYPE_CONFIGS=("f16" "q8_0" "q4_0" "tbq3_0" "tbq4_0" "pq3_0" "pq4_0")

MIXED_CONFIGS=(
    "tbq3_0:pq3_0"
    "tbq4_0:pq4_0"
    "tbq3_0:q8_0"
    "tbq4_0:f16"
)

extract_pp() {
    echo "$1" | grep -E '^\|' | grep -E '\bpp[0-9]' | awk -F'|' '{n=NF; gsub(/ /, "", $(n-1)); split($(n-1), a, "±"); print a[1]}' | tail -1
}

extract_tg() {
    echo "$1" | grep -E '^\|' | grep -E '\btg[0-9]' | awk -F'|' '{n=NF; gsub(/ /, "", $(n-1)); split($(n-1), a, "±"); print a[1]}' | tail -1
}

run_bench() {
    local k_type="$1"
    local v_type="$2"
    local env_prefix="$3"
    local label="$4"

    local output
    output=$(env $env_prefix "$BENCH" \
        -m "$MODEL_PATH" \
        --cache-type-k "$k_type" \
        --cache-type-v "$v_type" \
        -fa 1 \
        -p "$PROMPT_LEN" \
        -n "$GEN_LEN" \
        -r "$REPS" \
        2>&1) || {
        echo "  FAILED: $label (K=$k_type V=$v_type)" >&2
        echo "$output" | tail -5 >&2
        return 1
    }

    if echo "$output" | grep -q "error:"; then
        echo "  FAILED: $label (K=$k_type V=$v_type) — $(echo "$output" | grep 'error:' | head -1)" >&2
        return 1
    fi

    local pp tg
    pp=$(extract_pp "$output")
    tg=$(extract_tg "$output")

    if [ -z "$pp" ] || [ -z "$tg" ]; then
        echo "  FAILED: $label — could not parse t/s from output" >&2
        return 1
    fi

    echo "$pp $tg"
}

# --- Print header ---

echo ""
echo "=========================================="
echo " KV Cache Quantization Performance Test"
echo "=========================================="
echo " Model:       $MODEL_NAME"
echo " Prompt len:  $PROMPT_LEN"
echo " Gen len:     $GEN_LEN"
echo " Reps:        $REPS"
echo "=========================================="
echo ""

# --- Same-type benchmarks ---

declare -A pp_on tg_on pp_off tg_off

echo "--- Same-type KV cache configs ---"
echo ""

for cache_type in "${SAME_TYPE_CONFIGS[@]}"; do
    echo "Running: K=$cache_type V=$cache_type (coopmat2=on) ..."
    result=$(run_bench "$cache_type" "$cache_type" "" "coopmat2=on") || continue
    pp_on["$cache_type"]=$(echo "$result" | awk '{print $1}')
    tg_on["$cache_type"]=$(echo "$result" | awk '{print $2}')
    echo "  pp=${pp_on[$cache_type]} t/s  tg=${tg_on[$cache_type]} t/s"
done

if [ "$SKIP_COOPMAT_COMPARE" != "1" ]; then
    echo ""
    for cache_type in "${SAME_TYPE_CONFIGS[@]}"; do
        echo "Running: K=$cache_type V=$cache_type (coopmat2=off) ..."
        result=$(run_bench "$cache_type" "$cache_type" "GGML_VK_DISABLE_COOPMAT2=1" "coopmat2=off") || continue
        pp_off["$cache_type"]=$(echo "$result" | awk '{print $1}')
        tg_off["$cache_type"]=$(echo "$result" | awk '{print $2}')
        echo "  pp=${pp_off[$cache_type]} t/s  tg=${tg_off[$cache_type]} t/s"
    done
fi

# --- Same-type summary ---

echo ""
echo "=========================================="
echo " Same-Type Results (t/s)"
echo "=========================================="

if [ "$SKIP_COOPMAT_COMPARE" != "1" ]; then
    printf "  %-10s %10s %10s %10s %10s %10s\n" \
        "Type" "pp(cm2)" "pp(scalar)" "pp speedup" "tg(cm2)" "tg(scalar)"
    printf "  %-10s %10s %10s %10s %10s %10s\n" \
        "----" "-------" "----------" "----------" "------" "---------"

    for cache_type in "${SAME_TYPE_CONFIGS[@]}"; do
        p_on="${pp_on[$cache_type]:-n/a}"
        p_off="${pp_off[$cache_type]:-n/a}"
        t_on="${tg_on[$cache_type]:-n/a}"
        t_off="${tg_off[$cache_type]:-n/a}"
        speedup="n/a"
        if [ "$p_on" != "n/a" ] && [ "$p_off" != "n/a" ]; then
            speedup=$(echo "$p_on $p_off" | awk '{if ($2 > 0) printf "%.2fx", $1/$2; else print "n/a"}')
        fi
        printf "  %-10s %10s %10s %10s %10s %10s\n" \
            "$cache_type" "$p_on" "$p_off" "$speedup" "$t_on" "$t_off"
    done
else
    printf "  %-10s %10s %10s\n" "Type" "pp" "tg"
    printf "  %-10s %10s %10s\n" "----" "--" "--"

    for cache_type in "${SAME_TYPE_CONFIGS[@]}"; do
        printf "  %-10s %10s %10s\n" \
            "$cache_type" "${pp_on[$cache_type]:-n/a}" "${tg_on[$cache_type]:-n/a}"
    done
fi

echo "=========================================="

# --- Mixed K/V benchmarks ---

declare -A mpp_on mtg_on mpp_off mtg_off

echo ""
echo "--- Mixed K/V cache configs ---"
echo ""

for mixed in "${MIXED_CONFIGS[@]}"; do
    k_type="${mixed%%:*}"
    v_type="${mixed##*:}"

    echo "Running: K=$k_type V=$v_type (coopmat2=on) ..."
    result=$(run_bench "$k_type" "$v_type" "" "coopmat2=on") || continue
    mpp_on["$mixed"]=$(echo "$result" | awk '{print $1}')
    mtg_on["$mixed"]=$(echo "$result" | awk '{print $2}')
    echo "  pp=${mpp_on[$mixed]} t/s  tg=${mtg_on[$mixed]} t/s"
done

if [ "$SKIP_COOPMAT_COMPARE" != "1" ]; then
    echo ""
    for mixed in "${MIXED_CONFIGS[@]}"; do
        k_type="${mixed%%:*}"
        v_type="${mixed##*:}"

        echo "Running: K=$k_type V=$v_type (coopmat2=off) ..."
        result=$(run_bench "$k_type" "$v_type" "GGML_VK_DISABLE_COOPMAT2=1" "coopmat2=off") || continue
        mpp_off["$mixed"]=$(echo "$result" | awk '{print $1}')
        mtg_off["$mixed"]=$(echo "$result" | awk '{print $2}')
        echo "  pp=${mpp_off[$mixed]} t/s  tg=${mtg_off[$mixed]} t/s"
    done
fi

# --- Mixed summary ---

echo ""
echo "=========================================="
echo " Mixed K/V Results (t/s)"
echo "=========================================="

if [ "$SKIP_COOPMAT_COMPARE" != "1" ]; then
    printf "  %-10s %-10s %10s %10s %10s %10s %10s\n" \
        "K type" "V type" "pp(cm2)" "pp(scalar)" "pp speedup" "tg(cm2)" "tg(scalar)"
    printf "  %-10s %-10s %10s %10s %10s %10s %10s\n" \
        "------" "------" "-------" "----------" "----------" "------" "---------"

    for mixed in "${MIXED_CONFIGS[@]}"; do
        k_type="${mixed%%:*}"
        v_type="${mixed##*:}"
        p_on="${mpp_on[$mixed]:-n/a}"
        p_off="${mpp_off[$mixed]:-n/a}"
        t_on="${mtg_on[$mixed]:-n/a}"
        t_off="${mtg_off[$mixed]:-n/a}"
        speedup="n/a"
        if [ "$p_on" != "n/a" ] && [ "$p_off" != "n/a" ]; then
            speedup=$(echo "$p_on $p_off" | awk '{if ($2 > 0) printf "%.2fx", $1/$2; else print "n/a"}')
        fi
        printf "  %-10s %-10s %10s %10s %10s %10s %10s\n" \
            "$k_type" "$v_type" "$p_on" "$p_off" "$speedup" "$t_on" "$t_off"
    done
else
    printf "  %-10s %-10s %10s %10s\n" "K type" "V type" "pp" "tg"
    printf "  %-10s %-10s %10s %10s\n" "------" "------" "--" "--"

    for mixed in "${MIXED_CONFIGS[@]}"; do
        k_type="${mixed%%:*}"
        v_type="${mixed##*:}"
        printf "  %-10s %-10s %10s %10s\n" \
            "$k_type" "$v_type" "${mpp_on[$mixed]:-n/a}" "${mtg_on[$mixed]:-n/a}"
    done
fi

echo "=========================================="
echo ""
echo "Done. pp = prefill (N>1, coopmat2 active), tg = decode (N=1, scalar FA)."
echo "Speedup > 1.0x in pp column indicates coopmat2 benefit."
