SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_config.sh"

load_model_profile "inference_eval" || exit 1
load_inference_eval_hparams
validate_model_params || exit 1

pids=()

METHODS=(
    "zero_shot"
    "few_shot"
)

OPTIMIZE_METHODS=(
    # "base"
    "no_optimize"
)

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for optimize_method in "${OPTIMIZE_METHODS[@]}"; do
            for seed_prompt in "${SEED_PROMPTS[@]}"; do
                for i in "${!MODEL_NAMES[@]}"; do
                    MODEL_NAME="${MODEL_NAMES[$i]}"
                    SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
                    PARAMS="${PARAMS_LIST[$i]}"
                    echo "Starting evaluation for model: $MODEL_NAME with params: $PARAMS, dataset: $dataset, method: $method, optimize_method: $optimize_method, seed_prompt: $seed_prompt"
                    nohup python3 evaluate.py --model_name "$MODEL_NAME" --api_params "$PARAMS" --dataset "$dataset" --method "$method" --optimize_method "$optimize_method" --seed_prompt "$seed_prompt" --trial "$TRIAL" > "logs/${method}_${dataset}_${seed_prompt}_${SANITIZED_MODEL_NAME}_trial${TRIAL}.log" 2>&1 &
                    pids+=($!)
                done
            done
            # wait for all background processes to finish
            for pid in "${pids[@]}"; do
                wait $pid
            done
        done
    done
done
