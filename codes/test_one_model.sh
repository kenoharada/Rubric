SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_config.sh"

load_model_profile "test_one_model" || exit 1
validate_model_params || exit 1

pids=()

OPTIMIZE_METHODS=(
    "base"
)

WITH_RATIONALES=(
    "True"
    "False"
)

ITERATIONS=(
    "3"
    "1"
)

TOP_KS=(
    "3"
)

TRAIN_SIZES=(
    # "4"
    # "8"
    "20"
    # "50"
    # "100"
)
TRIAL="0"

DATASETS=(
    "asap_1"
    # "ets"
)

SEED_PROMPTS=(
    "expert"
    # "simplest"
    # "simple"
)

for dataset in "${DATASETS[@]}"; do
    for method in "${OPTIMIZE_METHODS[@]}"; do
        for seed_prompt in "${SEED_PROMPTS[@]}"; do
            for with_rationale in "${WITH_RATIONALES[@]}"; do
                for iteration in "${ITERATIONS[@]}"; do
                    for top_k in "${TOP_KS[@]}"; do
                        for train_size in "${TRAIN_SIZES[@]}"; do
                            for i in "${!MODEL_NAMES[@]}"; do
                                MODEL_NAME="${MODEL_NAMES[$i]}"
                                SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
                                PARAMS="${PARAMS_LIST[$i]}"
                                echo "Starting optimization for model: $MODEL_NAME with params: $PARAMS, dataset: $dataset, method: $method, seed_prompt: $seed_prompt, with_rationale: $with_rationale, iteration: $iteration, top_k: $top_k, train_size: $train_size, trial: $TRIAL"
                                nohup python3 inference.py \
                                    --optimize_method "$method" \
                                    --with_rationale "$with_rationale" \
                                    --iteration "$iteration" \
                                    --trial "$TRIAL" \
                                    --top_k "$top_k" \
                                    --seed_prompt "$seed_prompt" \
                                    --model_name "$MODEL_NAME" \
                                    --dataset "$dataset" \
                                    --api_params "$PARAMS" \
                                    --train_size "$train_size" \
                                    > "logs/${method}_${dataset}_${seed_prompt}_${SANITIZED_MODEL_NAME}_trial${TRIAL}.log" 2>&1 &
                                pids+=($!)
                            done
                            # wait for all background processes to finish
                            for pid in "${pids[@]}"; do
                                wait $pid
                            done
                            pids=()
                        done
                    done
                done
            done
        done
    done
done
