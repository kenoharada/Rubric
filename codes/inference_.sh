# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_name', type=str, default='gpt-4.1')
# parser.add_argument('--api_params', type=str, default='{}')
# args = parser.parse_args()
# asyncio.run(main(args))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_config.sh"

load_model_profile "inference_plain" || exit 1
validate_model_params || exit 1

pids=()

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    PARAMS="${PARAMS_LIST[$i]}"
    echo "Starting inference for model: $MODEL_NAME with params: $PARAMS"
    nohup python3 inference.py --model_name "$MODEL_NAME" --api_params "$PARAMS" > "inference_${MODEL_NAME}.log" 2>&1 &
    pids+=($!)
done

# Wait for all background processes to finish
for pid in "${pids[@]}"; do
    wait $pid
done
