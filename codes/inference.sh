SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_config.sh"

load_model_profile "inference_eval" || exit 1
load_inference_eval_hparams
validate_model_params || exit 1

rm logs/*.log
pids=()

OPTIMIZE_METHODS=(
    "base"
)

for dataset in "${DATASETS[@]}"; do
    for method in "${OPTIMIZE_METHODS[@]}"; do
        for seed_prompt in "${SEED_PROMPTS[@]}"; do
            for with_rationale in "${WITH_RATIONALES[@]}"; do
                for iteration in "${ITERATIONS[@]}"; do
                    for top_k in "${TOP_KS[@]}"; do
                        for train_size in "${TRAIN_SIZES[@]}"; do
                            for batch_sizes in "${BATCH_SIZE_SETS[@]}"; do
                                for monte_carlo_runs in "${MONTE_CARLO_RUNS[@]}"; do
                                    read -r -a batch_size_array <<< "$batch_sizes"
                                    batch_sizes_tag=$(echo "$batch_sizes" | tr ' ' '-')
                                    for i in "${!MODEL_NAMES[@]}"; do
                                        MODEL_NAME="${MODEL_NAMES[$i]}"
                                        SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
                                        PARAMS="${PARAMS_LIST[$i]}"
                                        # if with_rationale == True and Iteration == 1, skip
                                        # if [[ "$with_rationale" == "True" && "$iteration" == "1" ]]; then
                                        #     continue
                                        # fi
                                        # if with_rationale == False and Iteration > 1, skip
                                        # if [[ "$with_rationale" == "False" && "$iteration" != "1" ]]; then
                                        #     continue
                                        # fi
                                        echo "Starting optimization for model: $MODEL_NAME with params: $PARAMS, dataset: $dataset, method: $method, seed_prompt: $seed_prompt, with_rationale: $with_rationale, iteration: $iteration, top_k: $top_k, train_size: $train_size, batch_sizes: [$batch_sizes], monte_carlo_runs: $monte_carlo_runs, trial: $TRIAL"
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
                                            --batch_sizes "${batch_size_array[@]}" \
                                            --monte_carlo_runs "$monte_carlo_runs" \
                                            > "logs/${method}_${dataset}_${seed_prompt}_${with_rationale}_train${train_size}_iteration${iteration}_top${top_k}_bs${batch_sizes_tag}_mc${monte_carlo_runs}_${SANITIZED_MODEL_NAME}_trial${TRIAL}.log" 2>&1 &
                                        pids+=($!)
                                    done
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
    done
done
