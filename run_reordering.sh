#!/bin/bash
set -e

# Activate the conda environment
source activate SpMM

# Directory containing matrices
ROOT_DIR="./data"

# Testing mode - set to true to process only one matrix for testing
TESTING_MODE=false

# Reordering methods
PARTITION_METHODS=("None" "metis" "louvain")
REORDER_METHODS=("rcm")
REORDER_LOCS=("local" "global")
RES_LIMITS=("1.0" "0.7" "1.3") # For Louvain and Leiden

# PARAMETERS
REPEAT=3

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Counter for processed directories
DIR_COUNT=0
TOTAL_COMBINATIONS=0

# Calculate total combinations for progress tracking
for PARTITION_METHOD in "${PARTITION_METHODS[@]}"; do
    for REORDER_METHOD in "${REORDER_METHODS[@]}"; do
        for REORDER_LOC in "${REORDER_LOCS[@]}"; do
            if [[ "$PARTITION_METHOD" == "None" && "$REORDER_LOC" == "local" ]]; then
                continue
            fi
            if [[ "$PARTITION_METHOD" == "louvain" ]]; then
                TOTAL_COMBINATIONS=$((TOTAL_COMBINATIONS + ${#RES_LIMITS[@]}))
            else
                TOTAL_COMBINATIONS=$((TOTAL_COMBINATIONS + 1))
            fi
        done
    done
done

echo "Total method combinations to test: $TOTAL_COMBINATIONS"

if [[ "$TESTING_MODE" == "true" ]]; then
    echo "⚠️  TESTING MODE: Will process only ONE matrix"
fi

# TODO: Don't process directories with already reordered matrices. In order to do so, record to a log file reordered matrices and check against it.

# Find all .mtx files in the structure
find "$ROOT_DIR" -type f -name "*.mtx" | while read -r MTX_PATH; do
    MATRIX_DIR=$(basename "$(dirname "$MTX_PATH")")
    MATRIX_FILE=$(basename "$MTX_PATH" .mtx)
    # Check if filename matches parent directory
    if [[ "$MATRIX_DIR" == "$MATRIX_FILE" ]]; then
        echo "Processing: $MTX_PATH"
        DIR_COUNT=$((DIR_COUNT + 1))
        
        # Create a subdirectory for this matrix's logs
        MATRIX_LOG_DIR="$LOG_DIR/$MATRIX_DIR"
        mkdir -p "$MATRIX_LOG_DIR"
        
        # Loop through all combinations of methods
        for PARTITION_METHOD in "${PARTITION_METHODS[@]}"; do
            for REORDER_METHOD in "${REORDER_METHODS[@]}"; do
                for REORDER_LOC in "${REORDER_LOCS[@]}"; do
                    # Skip local reordering when partition method is "None"
                    if [[ "$PARTITION_METHOD" == "None" && "$REORDER_LOC" == "local" ]]; then
                        continue
                    fi
                    
                    # For Louvain method, also loop through resolution limits
                    if [[ "$PARTITION_METHOD" == "louvain" ]]; then
                        for RES_LIMIT in "${RES_LIMITS[@]}"; do
                            METHOD_COMBO="${PARTITION_METHOD}_${REORDER_LOC}_${REORDER_METHOD}_res${RES_LIMIT}"
                            echo "  → Reordering with method: $METHOD_COMBO"
                            
                            if [[ "$PARTITION_METHOD" == "None" ]]; then
                                python reorder.py --file "$MTX_PATH" --reorder-method "$REORDER_METHOD" \
                                    --ordering-place "$REORDER_LOC" --repeats "$REPEAT" \
                                    >> "$MATRIX_LOG_DIR/$(basename "$MTX_PATH")_${METHOD_COMBO}.log" 2>&1
                            else
                                python reorder.py --file "$MTX_PATH" --partition-method "$PARTITION_METHOD" \
                                    --reorder-method "$REORDER_METHOD" --ordering-place "$REORDER_LOC" \
                                    --res_limit "$RES_LIMIT" --repeats "$REPEAT" \
                                    >> "$MATRIX_LOG_DIR/$(basename "$MTX_PATH")_${METHOD_COMBO}.log" 2>&1
                            fi
                            
                            # Check if the command was successful
                            if [[ $? -eq 0 ]]; then
                                echo "    ✓ Success"
                            else
                                echo "    ✗ Failed (check log: $(basename "$MTX_PATH")_${METHOD_COMBO}.log)"
                            fi
                        done
                    else
                        METHOD_COMBO="${PARTITION_METHOD}_${REORDER_LOC}_${REORDER_METHOD}"
                        echo "  → Reordering with method: $METHOD_COMBO"
                        
                        if [[ "$PARTITION_METHOD" == "None" ]]; then
                            python reorder.py --file "$MTX_PATH" --reorder-method "$REORDER_METHOD" \
                                --ordering-place "$REORDER_LOC" --repeats "$REPEAT" \
                                >> "$MATRIX_LOG_DIR/$(basename "$MTX_PATH")_${METHOD_COMBO}.log" 2>&1
                        else
                            python reorder.py --file "$MTX_PATH" --partition-method "$PARTITION_METHOD" \
                                --repeats "$REPEAT"\
                                --reorder-method "$REORDER_METHOD" --ordering-place "$REORDER_LOC" \
                                >> "$MATRIX_LOG_DIR/$(basename "$MTX_PATH")_${METHOD_COMBO}.log" 2>&1
                        fi
                        
                        # Check if the command was successful
                        if [[ $? -eq 0 ]]; then
                            echo "    ✓ Success"
                        else
                            echo "    ✗ Failed (check log: $(basename "$MTX_PATH")_${METHOD_COMBO}.log)"
                        fi
                    fi
                done
            done
        done
        
        # Exit after processing first matrix if in testing mode
        if [[ "$TESTING_MODE" == "true" ]]; then
            echo "🛑 Testing mode: Stopping after first matrix"
            break
        fi
    else
        echo "Skipping: $MTX_PATH (filename ≠ directory)"
    fi
done

echo "=================================================="
echo "Processing complete!"
echo "Total directories processed: $DIR_COUNT"
echo "Total method combinations tested: $TOTAL_COMBINATIONS"
echo "Log files can be found in: $LOG_DIR"
echo "=================================================="
