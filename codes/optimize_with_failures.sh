SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_config.sh"

load_model_profile "legacy" || exit 1
validate_model_params || exit 1

pids=()

DATSETS=(
    "asap_1"
    "ets"
)
for DATASET in "${DATSETS[@]}"; do
    
    for i in "${!MODEL_NAMES[@]}"; do
        MODEL_NAME="${MODEL_NAMES[$i]}"
        SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
        PARAMS="${PARAMS_LIST[$i]}"
        echo "Starting optimized for model: $MODEL_NAME with params: $PARAMS and datasets: $DATASET"
        nohup python3 optimize_with_failures.py --model_name "$MODEL_NAME" --api_params "$PARAMS" --dataset "$DATASET" > "logs/optimized_${DATASET}_${SANITIZED_MODEL_NAME}.log" 2>&1 &
        pids+=($!)
    done
    
    for pid in "${pids[@]}"; do
        wait $pid
    done
done

# Wait for all background processes to finish
for pid in "${pids[@]}"; do
    wait $pid
done
