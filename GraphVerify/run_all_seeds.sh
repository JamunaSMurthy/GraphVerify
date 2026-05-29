#!/usr/bin/env bash
# ============================================================
# run_all_seeds.sh
# Runs GraphVerify over all datasets and seeds (Table 1 / Table 2)
#
# Usage:
#   ./run_all_seeds.sh [--llm_backend openai] [--llm_model gpt-4o-mini] \
#                      [--max_samples 500] [--graph_dir output/graphs]
# ============================================================

set -e

DATASETS="hotpotqa 2wikimultihopqa musique fever ragtruth"
SEEDS="0 1 2"
SPLIT="validation"
LLM_BACKEND="${LLM_BACKEND:-openai}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GRAPH_DIR="${GRAPH_DIR:-output/graphs}"
PRED_DIR="output/predictions"
RESULT_DIR="output/results"

echo "============================================================"
echo " GraphVerify: Full Evaluation Run"
echo " LLM: ${LLM_BACKEND} / ${LLM_MODEL}"
echo "============================================================"

# Step 1: Build graphs (if not already done)
echo ""
echo "=== Step 1: Build Evidence Graphs ==="
for DATASET in $DATASETS; do
    OUT="${GRAPH_DIR}/${DATASET}"
    mkdir -p "$OUT"
    CMD="python graph_build.py \
        --dataset $DATASET \
        --split $SPLIT \
        --output_dir $OUT \
        --llm_backend $LLM_BACKEND \
        --llm_model $LLM_MODEL"
    if [ -n "$MAX_SAMPLES" ]; then CMD="$CMD --max_samples $MAX_SAMPLES"; fi
    echo "  $DATASET..."
    eval "$CMD"
done

# Step 2: Run verification for each dataset × seed
echo ""
echo "=== Step 2: Run Verification ==="
for DATASET in $DATASETS; do
    for SEED in $SEEDS; do
        OUT="${PRED_DIR}/${DATASET}"
        mkdir -p "$OUT"
        CMD="python verify.py \
            --dataset $DATASET \
            --split $SPLIT \
            --graph_dir ${GRAPH_DIR}/${DATASET} \
            --output_dir $OUT \
            --llm_backend $LLM_BACKEND \
            --llm_model $LLM_MODEL \
            --seed $SEED"
        if [ -n "$MAX_SAMPLES" ]; then CMD="$CMD --max_samples $MAX_SAMPLES"; fi
        echo "  $DATASET seed=$SEED..."
        eval "$CMD"
    done
done

# Step 3: Calibrate on validation split (seed 0)
echo ""
echo "=== Step 3: Calibrate ==="
for DATASET in $DATASETS; do
    python calibrate.py \
        --pred_dir "${PRED_DIR}/${DATASET}" \
        --dataset $DATASET \
        --split $SPLIT \
        --seed 0 \
        --output_dir output/calibrators
    echo "  $DATASET calibrated."
done

# Step 4: Evaluate (Table 1 format)
echo ""
echo "=== Step 4: Evaluate ==="
mkdir -p "$RESULT_DIR"
python eval/evaluate.py \
    --pred_root "$PRED_DIR" \
    --datasets "$(echo $DATASETS | tr ' ' ',')" \
    --split "$SPLIT" \
    --seeds "$(echo $SEEDS | tr ' ' ',')" \
    --output "${RESULT_DIR}/table1.json"

echo ""
echo "Done. Results in ${RESULT_DIR}/table1.json"
