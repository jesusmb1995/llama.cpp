#!/usr/bin/env bash
#
# Benchmark KV cache quantization throughput (prefill pp & decode tg t/s).
# Auto-detects coopmat1/coopmat2 GPU support and compares cooperative matrix
# flash attention vs scalar fallback for TBQ/PQ quantizations.
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
#   FORCE_COOPMAT — override autodetect: "cm1", "cm2", "both", or "none"
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
FORCE_COOPMAT="${FORCE_COOPMAT:-}"

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

# --- Detect coopmat support ---

detect_coopmat() {
    if [ -n "$FORCE_COOPMAT" ]; then
        echo "$FORCE_COOPMAT"
        return
    fi

    if ! command -v vulkaninfo &> /dev/null; then
        echo "Warning: vulkaninfo not found, cannot autodetect coopmat. Assuming none." >&2
        echo "none"
        return
    fi

    local vk_extensions
    vk_extensions=$(vulkaninfo 2>/dev/null || true)

    if [ -z "$vk_extensions" ]; then
        echo "Warning: vulkaninfo produced no output, assuming no coopmat support." >&2
        echo "none"
        return
    fi

    local has_cm1=0 has_cm2=0
    grep -q "VK_KHR_cooperative_matrix" <<< "$vk_extensions" && has_cm1=1
    grep -q "VK_NV_cooperative_matrix2" <<< "$vk_extensions" && has_cm2=1

    if [ "$has_cm1" = "1" ] && [ "$has_cm2" = "1" ]; then
        echo "both"
    elif [ "$has_cm2" = "1" ]; then
        echo "cm2"
    elif [ "$has_cm1" = "1" ]; then
        echo "cm1"
    else
        echo "none"
    fi
}

COOPMAT_LEVEL=$(detect_coopmat)

# Build ordered list of coopmat levels to benchmark.
# Each level has a label and the env var used to *run with only that level*.
# "scalar" is always the fallback baseline when comparing.
COOPMAT_LEVELS=()

case "$COOPMAT_LEVEL" in
    both)
        COOPMAT_LEVELS=("cm1" "cm2")
        ;;
    cm2)
        COOPMAT_LEVELS=("cm2")
        ;;
    cm1)
        COOPMAT_LEVELS=("cm1")
        ;;
    none)
        COOPMAT_LEVELS=()
        SKIP_COOPMAT_COMPARE=1
        ;;
    *)
        echo "Error: unknown FORCE_COOPMAT value '$COOPMAT_LEVEL' (expected cm1, cm2, both, or none)"
        exit 1
        ;;
esac

coopmat_label_for() {
    case "$1" in
        cm1) echo "coopmat1" ;;
        cm2) echo "coopmat2" ;;
        *)   echo "scalar"  ;;
    esac
}

# Env prefix to run with *only* a specific coopmat level enabled.
# cm1: disable cm2 so only cm1 is active
# cm2: default (cm2 takes precedence when both are available)
# scalar: disable all coopmat
coopmat_env_for() {
    case "$1" in
        cm1) echo "GGML_VK_DISABLE_COOPMAT2=1" ;;
        cm2) echo "" ;;
        *)   echo "GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1" ;;
    esac
}

# Env to disable all coopmat (scalar baseline)
SCALAR_DISABLE_ENV="GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1"

# For display: top-level label
if [ ${#COOPMAT_LEVELS[@]} -gt 0 ]; then
    COOPMAT_DISPLAY=$(printf '%s+' "${COOPMAT_LEVELS[@]}")
    COOPMAT_DISPLAY="${COOPMAT_DISPLAY%+}"
else
    COOPMAT_DISPLAY="none"
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
echo " GPU coopmat: $COOPMAT_LEVEL ($COOPMAT_DISPLAY)"
if [ "$SKIP_COOPMAT_COMPARE" = "1" ]; then
    echo " Compare:     disabled"
else
    for lvl in "${COOPMAT_LEVELS[@]}"; do
        echo " Compare:     $(coopmat_label_for "$lvl") vs scalar"
    done
fi
echo "=========================================="
echo ""

# --- Same-type benchmarks ---

# Per-level results: pp_<level>[config], tg_<level>[config]
# Scalar baseline:   pp_scalar[config], tg_scalar[config]
declare -A pp_scalar tg_scalar

echo "--- Same-type KV cache configs ---"
echo ""

# Run each coopmat level
for lvl in "${COOPMAT_LEVELS[@]}"; do
    declare -A "pp_${lvl}" "tg_${lvl}"
    local_label=$(coopmat_label_for "$lvl")
    local_env=$(coopmat_env_for "$lvl")

    for cache_type in "${SAME_TYPE_CONFIGS[@]}"; do
        echo "Running: K=$cache_type V=$cache_type ($local_label) ..."
        result=$(run_bench "$cache_type" "$cache_type" "$local_env" "$local_label") || continue
        eval "pp_${lvl}[\"$cache_type\"]=$(echo "$result" | awk '{print $1}')"
        eval "tg_${lvl}[\"$cache_type\"]=$(echo "$result" | awk '{print $2}')"
        eval "echo \"  pp=\${pp_${lvl}[$cache_type]} t/s  tg=\${tg_${lvl}[$cache_type]} t/s\""
    done
    echo ""
done

# When no coopmat levels, run a plain (scalar) benchmark
if [ ${#COOPMAT_LEVELS[@]} -eq 0 ]; then
    for cache_type in "${SAME_TYPE_CONFIGS[@]}"; do
        echo "Running: K=$cache_type V=$cache_type (scalar) ..."
        result=$(run_bench "$cache_type" "$cache_type" "" "scalar") || continue
        pp_scalar["$cache_type"]=$(echo "$result" | awk '{print $1}')
        tg_scalar["$cache_type"]=$(echo "$result" | awk '{print $2}')
        echo "  pp=${pp_scalar[$cache_type]} t/s  tg=${tg_scalar[$cache_type]} t/s"
    done
    echo ""
fi

# Run scalar baseline (all coopmat disabled) for comparison
if [ "$SKIP_COOPMAT_COMPARE" != "1" ]; then
    for cache_type in "${SAME_TYPE_CONFIGS[@]}"; do
        echo "Running: K=$cache_type V=$cache_type (scalar) ..."
        result=$(run_bench "$cache_type" "$cache_type" "$SCALAR_DISABLE_ENV" "scalar") || continue
        pp_scalar["$cache_type"]=$(echo "$result" | awk '{print $1}')
        tg_scalar["$cache_type"]=$(echo "$result" | awk '{print $2}')
        echo "  pp=${pp_scalar[$cache_type]} t/s  tg=${tg_scalar[$cache_type]} t/s"
    done
fi

# --- Same-type summary ---

print_same_type_table() {
    local lvl="$1"
    local cm_tag
    cm_tag=$(coopmat_label_for "$lvl")

    echo ""
    echo "=========================================="
    echo " Same-Type Results: $cm_tag vs scalar (t/s)"
    echo "=========================================="

    printf "  %-10s %10s %10s %10s %10s %10s\n" \
        "Type" "pp($cm_tag)" "pp(scalar)" "pp speedup" "tg($cm_tag)" "tg(scalar)"
    printf "  %-10s %10s %10s %10s %10s %10s\n" \
        "----" "-------" "----------" "----------" "------" "---------"

    for cache_type in "${SAME_TYPE_CONFIGS[@]}"; do
        local p_on p_off t_on t_off speedup
        eval "p_on=\${pp_${lvl}[$cache_type]:-n/a}"
        p_off="${pp_scalar[$cache_type]:-n/a}"
        eval "t_on=\${tg_${lvl}[$cache_type]:-n/a}"
        t_off="${tg_scalar[$cache_type]:-n/a}"
        speedup="n/a"
        if [ "$p_on" != "n/a" ] && [ "$p_off" != "n/a" ]; then
            speedup=$(echo "$p_on $p_off" | awk '{if ($2 > 0) printf "%.2fx", $1/$2; else print "n/a"}')
        fi
        printf "  %-10s %10s %10s %10s %10s %10s\n" \
            "$cache_type" "$p_on" "$p_off" "$speedup" "$t_on" "$t_off"
    done

    echo "=========================================="
}

if [ "$SKIP_COOPMAT_COMPARE" != "1" ]; then
    for lvl in "${COOPMAT_LEVELS[@]}"; do
        print_same_type_table "$lvl"
    done
else
    echo ""
    echo "=========================================="
    echo " Same-Type Results (t/s)"
    echo "=========================================="

    printf "  %-10s %10s %10s\n" "Type" "pp" "tg"
    printf "  %-10s %10s %10s\n" "----" "--" "--"

    for cache_type in "${SAME_TYPE_CONFIGS[@]}"; do
        printf "  %-10s %10s %10s\n" \
            "$cache_type" "${pp_scalar[$cache_type]:-n/a}" "${tg_scalar[$cache_type]:-n/a}"
    done

    echo "=========================================="
fi

# --- Mixed K/V benchmarks ---

declare -A mpp_scalar mtg_scalar

echo ""
echo "--- Mixed K/V cache configs ---"
echo ""

# Run each coopmat level
for lvl in "${COOPMAT_LEVELS[@]}"; do
    declare -A "mpp_${lvl}" "mtg_${lvl}"
    local_label=$(coopmat_label_for "$lvl")
    local_env=$(coopmat_env_for "$lvl")

    for mixed in "${MIXED_CONFIGS[@]}"; do
        k_type="${mixed%%:*}"
        v_type="${mixed##*:}"

        echo "Running: K=$k_type V=$v_type ($local_label) ..."
        result=$(run_bench "$k_type" "$v_type" "$local_env" "$local_label") || continue
        eval "mpp_${lvl}[\"$mixed\"]=$(echo "$result" | awk '{print $1}')"
        eval "mtg_${lvl}[\"$mixed\"]=$(echo "$result" | awk '{print $2}')"
        eval "echo \"  pp=\${mpp_${lvl}[$mixed]} t/s  tg=\${mtg_${lvl}[$mixed]} t/s\""
    done
    echo ""
done

# When no coopmat levels, run plain (scalar) benchmark
if [ ${#COOPMAT_LEVELS[@]} -eq 0 ]; then
    for mixed in "${MIXED_CONFIGS[@]}"; do
        k_type="${mixed%%:*}"
        v_type="${mixed##*:}"

        echo "Running: K=$k_type V=$v_type (scalar) ..."
        result=$(run_bench "$k_type" "$v_type" "" "scalar") || continue
        mpp_scalar["$mixed"]=$(echo "$result" | awk '{print $1}')
        mtg_scalar["$mixed"]=$(echo "$result" | awk '{print $2}')
        echo "  pp=${mpp_scalar[$mixed]} t/s  tg=${mtg_scalar[$mixed]} t/s"
    done
    echo ""
fi

# Run scalar baseline (all coopmat disabled) for comparison
if [ "$SKIP_COOPMAT_COMPARE" != "1" ]; then
    for mixed in "${MIXED_CONFIGS[@]}"; do
        k_type="${mixed%%:*}"
        v_type="${mixed##*:}"

        echo "Running: K=$k_type V=$v_type (scalar) ..."
        result=$(run_bench "$k_type" "$v_type" "$SCALAR_DISABLE_ENV" "scalar") || continue
        mpp_scalar["$mixed"]=$(echo "$result" | awk '{print $1}')
        mtg_scalar["$mixed"]=$(echo "$result" | awk '{print $2}')
        echo "  pp=${mpp_scalar[$mixed]} t/s  tg=${mtg_scalar[$mixed]} t/s"
    done
fi

# --- Mixed summary ---

print_mixed_table() {
    local lvl="$1"
    local cm_tag
    cm_tag=$(coopmat_label_for "$lvl")

    echo ""
    echo "=========================================="
    echo " Mixed K/V Results: $cm_tag vs scalar (t/s)"
    echo "=========================================="

    printf "  %-10s %-10s %10s %10s %10s %10s %10s\n" \
        "K type" "V type" "pp($cm_tag)" "pp(scalar)" "pp speedup" "tg($cm_tag)" "tg(scalar)"
    printf "  %-10s %-10s %10s %10s %10s %10s %10s\n" \
        "------" "------" "-------" "----------" "----------" "------" "---------"

    for mixed in "${MIXED_CONFIGS[@]}"; do
        k_type="${mixed%%:*}"
        v_type="${mixed##*:}"
        local p_on p_off t_on t_off speedup
        eval "p_on=\${mpp_${lvl}[$mixed]:-n/a}"
        p_off="${mpp_scalar[$mixed]:-n/a}"
        eval "t_on=\${mtg_${lvl}[$mixed]:-n/a}"
        t_off="${mtg_scalar[$mixed]:-n/a}"
        speedup="n/a"
        if [ "$p_on" != "n/a" ] && [ "$p_off" != "n/a" ]; then
            speedup=$(echo "$p_on $p_off" | awk '{if ($2 > 0) printf "%.2fx", $1/$2; else print "n/a"}')
        fi
        printf "  %-10s %-10s %10s %10s %10s %10s %10s\n" \
            "$k_type" "$v_type" "$p_on" "$p_off" "$speedup" "$t_on" "$t_off"
    done

    echo "=========================================="
}

if [ "$SKIP_COOPMAT_COMPARE" != "1" ]; then
    for lvl in "${COOPMAT_LEVELS[@]}"; do
        print_mixed_table "$lvl"
    done
else
    echo ""
    echo "=========================================="
    echo " Mixed K/V Results (t/s)"
    echo "=========================================="

    printf "  %-10s %-10s %10s %10s\n" "K type" "V type" "pp" "tg"
    printf "  %-10s %-10s %10s %10s\n" "------" "------" "--" "--"

    for mixed in "${MIXED_CONFIGS[@]}"; do
        k_type="${mixed%%:*}"
        v_type="${mixed##*:}"
        printf "  %-10s %-10s %10s %10s\n" \
            "$k_type" "$v_type" "${mpp_scalar[$mixed]:-n/a}" "${mtg_scalar[$mixed]:-n/a}"
    done

    echo "=========================================="
fi

echo ""
echo "Done. Detected coopmat level: $COOPMAT_LEVEL."
if [ "$SKIP_COOPMAT_COMPARE" != "1" ]; then
    for lvl in "${COOPMAT_LEVELS[@]}"; do
        local_label=$(coopmat_label_for "$lvl")
        echo "Tables compare $local_label (prefill, N>1) vs scalar fallback."
    done
    echo "Speedup > 1.0x indicates coopmat benefit."
else
    echo "No coopmat comparison (either no coopmat support or SKIP_COOPMAT_COMPARE=1)."
fi
