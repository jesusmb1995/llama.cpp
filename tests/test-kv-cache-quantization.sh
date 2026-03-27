#!/usr/bin/env bash
#
# Test KV cache quantization quality by comparing perplexity across cache types.
# Downloads a small model and wikitext dataset if not already present.
#
# Usage:
#   tests/test-kv-cache-quantization.sh [build_dir]
#
# Example:
#   tests/test-kv-cache-quantization.sh build
#
set -euo pipefail

BUILD_DIR="${1:-build}"
PERPLEXITY="$BUILD_DIR/bin/llama-perplexity"

MODEL_DIR="${MODEL_DIR:-models}"
# Default: Mistral-7B-Instruct-v0.3 (head_dim=128, Apache license, ~4.1GB)
MODEL_NAME="${MODEL_NAME:-Mistral-7B-Instruct-v0.3-Q4_K_S.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_S.gguf}"
# Override via env for other models, e.g. head_dim=64:
#   MODEL_NAME=Llama-3.2-1B-Instruct-Q4_0.gguf \
#   MODEL_URL=https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf \
#   tests/test-kv-cache-quantization.sh build-tq
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

DATASET_DIR="wikitext-2-raw"
DATASET_FILE="$DATASET_DIR/wiki.test.raw"
DATASET_ZIP="wikitext-2-raw-v1.zip"
DATASET_URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/$DATASET_ZIP"

# Use fewer chunks for a faster smoke test; override with full dataset for CI
N_CHUNKS="${N_CHUNKS:-4}"
N_CTX="${N_CTX:-128}"

# Maximum allowed perplexity regression vs f16 baseline (percentage)
MAX_PPL_REGRESSION_PCT="${MAX_PPL_REGRESSION_PCT:-21}"

CACHE_TYPES=("tq3_0" "q4_0" "tq4_0" "f16" "q8_0")

# --- Download dependencies ---

if [ ! -f "$PERPLEXITY" ]; then
    echo "Error: $PERPLEXITY not found. Build the project first."
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

    # Sanity check: GGUF files are at least a few MB
    FILE_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -lt 1000000 ]; then
        echo "Error: downloaded file is only $FILE_SIZE bytes — likely a redirect or error page."
        echo "Download the model manually:"
        echo "  curl -L --fail -o $MODEL_PATH $MODEL_URL"
        rm -f "$MODEL_PATH"
        exit 1
    fi
    echo "Model downloaded to $MODEL_PATH ($((FILE_SIZE / 1048576)) MB)"
else
    echo "Model already exists: $MODEL_PATH"
fi

if [ ! -f "$DATASET_FILE" ]; then
    echo "Downloading wikitext-2 dataset..."
    if command -v curl &> /dev/null; then
        curl -L -o "$DATASET_ZIP" "$DATASET_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$DATASET_ZIP" "$DATASET_URL"
    else
        echo "Error: neither curl nor wget found"
        exit 1
    fi
    unzip -o "$DATASET_ZIP"
    rm -f "$DATASET_ZIP"
    echo "Dataset extracted to $DATASET_DIR"
else
    echo "Dataset already exists: $DATASET_FILE"
fi

# --- Run perplexity for each cache type ---

declare -A ppl_results

extract_ppl() {
    # Parse "Final estimate: PPL = 12.3456 +/- 0.12345" from output
    local output="$1"
    echo "$output" | grep -oP 'PPL = \K[0-9]+\.[0-9]+' | tail -1
}

echo ""
echo "=========================================="
echo " KV Cache Quantization Perplexity Test"
echo "=========================================="
echo " Model:    $MODEL_NAME"
echo " Dataset:  $DATASET_FILE"
echo " Chunks:   $N_CHUNKS"
echo " Context:  $N_CTX"
echo "=========================================="
echo ""

declare -A time_results

for cache_type in "${CACHE_TYPES[@]}"; do
    echo "--- Running perplexity with --cache-type-k $cache_type --cache-type-v $cache_type ---"

    start_time=$(date +%s%N)

    output=$("$PERPLEXITY" \
        -m "$MODEL_PATH" \
        -f "$DATASET_FILE" \
        --cache-type-k "$cache_type" \
        --cache-type-v "$cache_type" \
        -n "$N_CTX" \
        --chunks "$N_CHUNKS" \
        2>&1) || {
        echo "FAILED: perplexity run crashed for cache type $cache_type"
        echo "$output" | tail -20
        exit 1
    }

    end_time=$(date +%s%N)
    elapsed_ms=$(( (end_time - start_time) / 1000000 ))
    elapsed_s=$(echo "$elapsed_ms" | awk '{printf "%.2f", $1/1000}')

    ppl=$(extract_ppl "$output")
    if [ -z "$ppl" ]; then
        echo "FAILED: could not parse PPL from output for cache type $cache_type"
        echo "$output" | tail -20
        exit 1
    fi

    ppl_results["$cache_type"]="$ppl"
    time_results["$cache_type"]="$elapsed_s"
    echo "  $cache_type PPL = $ppl  (${elapsed_s}s)"
    echo ""
done

# --- Print summary table ---

echo ""
echo "=========================================="
echo " Results Summary"
echo "=========================================="
printf "  %-10s %10s %12s %10s\n" "Type" "PPL" "vs f16" "Time"
printf "  %-10s %10s %12s %10s\n" "----" "---" "------" "----"

baseline_ppl="${ppl_results[f16]}"
baseline_time="${time_results[f16]}"

for cache_type in "${CACHE_TYPES[@]}"; do
    ppl="${ppl_results[$cache_type]}"
    elapsed="${time_results[$cache_type]}"
    if [ "$cache_type" = "f16" ]; then
        printf "  %-10s %10s %12s %9ss\n" "$cache_type" "$ppl" "(baseline)" "$elapsed"
    else
        regression=$(echo "$ppl $baseline_ppl" | awk '{printf "%.2f", (($1 - $2) / $2) * 100}')
        slowdown=$(echo "$elapsed $baseline_time" | awk '{if ($2 > 0) printf "%.1fx", $1/$2; else print "n/a"}')
        printf "  %-10s %10s %+11s%% %8ss (%s)\n" "$cache_type" "$ppl" "$regression" "$elapsed" "$slowdown"
    fi
done

echo "=========================================="

# --- Check regressions ---

num_failed=0

for cache_type in "${CACHE_TYPES[@]}"; do
    [ "$cache_type" = "f16" ] && continue

    ppl="${ppl_results[$cache_type]}"
    exceeded=$(echo "$ppl $baseline_ppl $MAX_PPL_REGRESSION_PCT" | \
        awk '{regression = (($1 - $2) / $2) * 100; print (regression > $3) ? "1" : "0"}')

    if [ "$exceeded" = "1" ]; then
        regression=$(echo "$ppl $baseline_ppl" | awk '{printf "%.2f", (($1 - $2) / $2) * 100}')
        echo "FAILED: $cache_type PPL regression ${regression}% exceeds ${MAX_PPL_REGRESSION_PCT}% threshold"
        num_failed=$((num_failed + 1))
    fi
done

# TQ4 should have lower or equal PPL compared to TQ3 (more bits = better quality)
tq3_ppl="${ppl_results[tq3_0]}"
tq4_ppl="${ppl_results[tq4_0]}"
tq4_worse=$(echo "$tq4_ppl $tq3_ppl" | awk '{print ($1 > $2) ? "1" : "0"}')
if [ "$tq4_worse" = "1" ]; then
    echo "FAILED: tq4_0 PPL ($tq4_ppl) should be <= tq3_0 PPL ($tq3_ppl)"
    num_failed=$((num_failed + 1))
fi

echo ""
if [ "$num_failed" -eq 0 ]; then
    echo "All checks passed."
else
    echo "$num_failed check(s) failed."
    exit 0
fi
