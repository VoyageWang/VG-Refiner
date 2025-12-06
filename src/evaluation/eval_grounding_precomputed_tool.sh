#!/bin/bash
set -e

# --- Configuration ---
MODEL_TYPE="VG-refiner_evf"
# The local path to the LLM model you want to evaluate.
MODEL_PATH=${1:-"./model_weights/VG-Refiner"}

# The base path for the datasets.
BASE_DATA_PATH="./data/"
# List of datasets to evaluate on.
DATASETS=(
    "refcoco_testA"
    "refcoco_testB"
    "refcocop_testA"
    "refcocop_testB"
    "refcocog_test"
)

# Customize the GPUs to use for parallel evaluation.
GPU_ARRAY=(0 1 2 3 4 5 6 7)

# --- Script Logic ---
for TEST_NAME in "${DATASETS[@]}"; do
    TEST_DATA_PATH="${BASE_DATA_PATH}/${TEST_NAME}"

    echo "============================================="
    echo "Evaluating on dataset: $TEST_NAME"
    echo "Dataset path: $TEST_DATA_PATH"
    echo "============================================="

    # Define the output directory based on the model and dataset names.
    OUTPUT_PATH="./grounding_eval_results/${MODEL_TYPE}/${TEST_NAME}"
    
    NUM_PARTS=${#GPU_ARRAY[@]}
    
    # Create the output directory.
    mkdir -p $OUTPUT_PATH
    echo "Results will be saved in: $OUTPUT_PATH"
    
    # Run evaluation processes in parallel across the specified GPUs.
    for i in $(seq 0 $((NUM_PARTS-1))); do
        gpu_id=${GPU_ARRAY[$i]}
        process_idx=$i
        
        echo "Starting process ${process_idx} on GPU ${gpu_id} for ${TEST_NAME}..."
        
        # Set the visible GPU for the current process.
        export CUDA_VISIBLE_DEVICES=$gpu_id
        
        # Run the evaluation script in the background.
        (
            python src/evaluation/evaluation_grounding_precomputed_tool.py \
                --model_path $MODEL_PATH \
                --output_path $OUTPUT_PATH \
                --test_data_path $TEST_DATA_PATH \
                --idx $process_idx \
                --num_parts $NUM_PARTS \
                --batch_size 16 || { echo "1" > /tmp/process_status.$$; kill -TERM -$$; }
        ) &
    done
    
    # Wait for all background processes to complete.
    wait
    
    echo "All evaluation processes for ${TEST_NAME} finished."
    echo "Calculating final metrics for ${TEST_NAME}..."
    
    # Run the script to calculate and display the final metrics.
    python src/evaluation/calculate_grounding_metrics.py --output_dir $OUTPUT_PATH

    echo "---------------------------------------------"
    echo "Finished evaluation for ${TEST_NAME}"
    echo "---------------------------------------------"
    echo ""
done

echo "All datasets evaluated."