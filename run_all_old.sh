#!/bin/bash
set -e 

DATA_DIR="./data"
RESULT_DIR="./results"
CONFIG_FILE="config.ini"
EXEC="./build/runspECK"

# Ensure result directory exists
mkdir -p "$RESULT_DIR"

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

# Add 1 for the original method
TOTAL_COMBINATIONS=$((TOTAL_COMBINATIONS + 1))

echo "Total method combinations to test: $TOTAL_COMBINATIONS"

# Loop over each matrix directory inside data/
for MATRIX_DIR in "$DATA_DIR"/*; do
    [ -d "$MATRIX_DIR" ] || continue  # Skip non-directories

    MATRIX_NAME=$(basename "$MATRIX_DIR")
    echo "🔍 Processing matrix: $MATRIX_NAME"
    DIR_COUNT=$((DIR_COUNT + 1))

    # Create result directory for this matrix
    mkdir -p "$RESULT_DIR/$MATRIX_NAME"

    # First, process the original matrix
    MTX_FILE="$MATRIX_DIR/$MATRIX_NAME.mtx"
    if [ -f "$MTX_FILE" ]; then
        LOG_FILE="$RESULT_DIR/$MATRIX_NAME/$MATRIX_NAME.log"
        QDREP_FILE="$RESULT_DIR/$MATRIX_NAME/$MATRIX_NAME.qdrep"
        SQLITE_FILE="$RESULT_DIR/$MATRIX_NAME/$MATRIX_NAME"_sqlite
        NSYS_REP="$RESULT_DIR/$MATRIX_NAME/$MATRIX_NAME.nsys-rep"

        # Remove old files if they exist
        rm -f "$QDREP_FILE" "$SQLITE_FILE" "$NSYS_REP" "$LOG_FILE"

        echo "  ⚙️ Profiling: $MTX_FILE (original)"
        nsys profile --force-overwrite true -o "$QDREP_FILE" "$EXEC" "$MTX_FILE" "$CONFIG_FILE" > "$LOG_FILE"
        nsys export --force-overwrite true --type sqlite --output "$SQLITE_FILE" "$NSYS_REP"
    else
        echo "  ⚠️ Missing: $MTX_FILE (skipping original)"
    fi

    # Now process all reordered variants based on the method combinations
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
                        MTX_FILE="$MATRIX_DIR/reordered_${METHOD_COMBO}_${MATRIX_NAME}.mtx"
                        
                        if [ -f "$MTX_FILE" ]; then
                            LOG_FILE="$RESULT_DIR/$MATRIX_NAME/reordered_${METHOD_COMBO}_${MATRIX_NAME}.log"
                            QDREP_FILE="$RESULT_DIR/$MATRIX_NAME/reordered_${METHOD_COMBO}_${MATRIX_NAME}.qdrep"
                            SQLITE_FILE="$RESULT_DIR/$MATRIX_NAME/reordered_${METHOD_COMBO}_${MATRIX_NAME}"_sqlite
                            NSYS_REP="$RESULT_DIR/$MATRIX_NAME/reordered_${METHOD_COMBO}_${MATRIX_NAME}.nsys-rep"

                            # Remove old files if they exist
                            rm -f "$QDREP_FILE" "$SQLITE_FILE" "$NSYS_REP" "$LOG_FILE"

                            echo "  ⚙️ Profiling: $MTX_FILE with method $METHOD_COMBO"
                            nsys profile --force-overwrite true -o "$QDREP_FILE" "$EXEC" "$MTX_FILE" "$CONFIG_FILE" > "$LOG_FILE"
                            nsys export --force-overwrite true --type sqlite --output "$SQLITE_FILE" "$NSYS_REP"
                        else
                            echo "  ⚠️ Missing: $MTX_FILE (skipping)"
                        fi
                    done
                else
                    METHOD_COMBO="${PARTITION_METHOD}_${REORDER_LOC}_${REORDER_METHOD}"
                    MTX_FILE="$MATRIX_DIR/reordered_${METHOD_COMBO}_${MATRIX_NAME}.mtx"
                    
                    if [ -f "$MTX_FILE" ]; then
                        LOG_FILE="$RESULT_DIR/$MATRIX_NAME/reordered_${METHOD_COMBO}_${MATRIX_NAME}.log"
                        QDREP_FILE="$RESULT_DIR/$MATRIX_NAME/reordered_${METHOD_COMBO}_${MATRIX_NAME}.qdrep"
                        SQLITE_FILE="$RESULT_DIR/$MATRIX_NAME/reordered_${METHOD_COMBO}_${MATRIX_NAME}"_sqlite
                        NSYS_REP="$RESULT_DIR/$MATRIX_NAME/reordered_${METHOD_COMBO}_${MATRIX_NAME}.nsys-rep"

                        # Remove old files if they exist
                        rm -f "$QDREP_FILE" "$SQLITE_FILE" "$NSYS_REP" "$LOG_FILE"

                        echo "  ⚙️ Profiling: $MTX_FILE with method $METHOD_COMBO"
                        nsys profile --force-overwrite true -o "$QDREP_FILE" "$EXEC" "$MTX_FILE" "$CONFIG_FILE" > "$LOG_FILE"
                        nsys export --force-overwrite true --type sqlite --output "$SQLITE_FILE" "$NSYS_REP"
                    else
                        echo "  ⚠️ Missing: $MTX_FILE (skipping)"
                    fi
                fi
            done
        done
    done
done

echo "=================================================="
echo "Processing complete!"
echo "Total directories processed: $DIR_COUNT"
echo "Total method combinations tested: $TOTAL_COMBINATIONS"
echo "Results can be found in: $RESULT_DIR"
echo "=================================================="
