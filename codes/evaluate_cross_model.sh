SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_config.sh"

load_model_profile "inference_eval" || exit 1
load_inference_eval_hparams
validate_model_params || exit 1

# 評価時に使うモデル設定
EVAL_MODEL_NAMES=("${MODEL_NAMES[@]}")
EVAL_PARAMS_LIST=("${PARAMS_LIST[@]}")

# 最適化時に使ったモデル設定
OPT_MODEL_NAMES=("${MODEL_NAMES[@]}")
OPT_PARAMS_LIST=("${PARAMS_LIST[@]}")

if [[ ${#EVAL_MODEL_NAMES[@]} -ne ${#EVAL_PARAMS_LIST[@]} ]]; then
    echo "EVAL_MODEL_NAMES and EVAL_PARAMS_LIST length mismatch: ${#EVAL_MODEL_NAMES[@]} vs ${#EVAL_PARAMS_LIST[@]}" >&2
    exit 1
fi

if [[ ${#OPT_MODEL_NAMES[@]} -ne ${#OPT_PARAMS_LIST[@]} ]]; then
    echo "OPT_MODEL_NAMES and OPT_PARAMS_LIST length mismatch: ${#OPT_MODEL_NAMES[@]} vs ${#OPT_PARAMS_LIST[@]}" >&2
    exit 1
fi

pids=()
mkdir -p logs
rm -f logs/eval_cross_*.log

METHODS=(
    "zero_shot"
    # "few_shot"
)

OPTIMIZE_METHODS=(
    "base"
    # "no_optimize"
)

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for optimize_method in "${OPTIMIZE_METHODS[@]}"; do
            for seed_prompt in "${SEED_PROMPTS[@]}"; do
                for with_rationale in "${WITH_RATIONALES[@]}"; do
                    for iteration in "${ITERATIONS[@]}"; do
                        # inference.sh と同じ組み合わせのみ実行
                        if [[ "$with_rationale" == "True" && "$iteration" == "1" ]]; then
                            continue
                        fi
                        if [[ "$with_rationale" == "False" && "$iteration" != "1" ]]; then
                            continue
                        fi
                        for top_k in "${TOP_KS[@]}"; do
                            for train_size in "${TRAIN_SIZES[@]}"; do
                                for batch_sizes in "${BATCH_SIZE_SETS[@]}"; do
                                    for monte_carlo_runs in "${MONTE_CARLO_RUNS[@]}"; do
                                        read -r -a batch_size_array <<< "$batch_sizes"
                                        batch_sizes_tag=$(echo "$batch_sizes" | tr ' ' '-')

                                        for opt_i in "${!OPT_MODEL_NAMES[@]}"; do
                                            OPT_MODEL_NAME="${OPT_MODEL_NAMES[$opt_i]}"
                                            OPT_PARAMS="${OPT_PARAMS_LIST[$opt_i]}"
                                            OPT_MODEL_TAG=$(echo "$OPT_MODEL_NAME" | tr '/' '_')

                                            for eval_i in "${!EVAL_MODEL_NAMES[@]}"; do
                                                EVAL_MODEL_NAME="${EVAL_MODEL_NAMES[$eval_i]}"
                                                EVAL_PARAMS="${EVAL_PARAMS_LIST[$eval_i]}"
                                                EVAL_MODEL_TAG=$(echo "$EVAL_MODEL_NAME" | tr '/' '_')

                                                # クロスモデル評価のみ実施（同一モデル同士は除外）
                                                if [[ "$EVAL_MODEL_NAME" == "$OPT_MODEL_NAME" ]]; then
                                                    continue
                                                fi

                                                echo "Starting cross-model evaluation: eval_model=$EVAL_MODEL_NAME, opt_model=$OPT_MODEL_NAME, dataset=$dataset, method=$method, optimize_method=$optimize_method, seed_prompt=$seed_prompt, with_rationale=$with_rationale, iteration=$iteration, top_k=$top_k, train_size=$train_size, batch_sizes=[$batch_sizes], monte_carlo_runs=$monte_carlo_runs, trial=$TRIAL"
                                                nohup python3 evaluate_cross_model.py \
                                                    --eval_model_name "$EVAL_MODEL_NAME" \
                                                    --eval_api_params "$EVAL_PARAMS" \
                                                    --opt_model_name "$OPT_MODEL_NAME" \
                                                    --opt_api_params "$OPT_PARAMS" \
                                                    --dataset "$dataset" \
                                                    --method "$method" \
                                                    --optimize_method "$optimize_method" \
                                                    --seed_prompt "$seed_prompt" \
                                                    --trial "$TRIAL" \
                                                    --with_rationale "$with_rationale" \
                                                    --iteration "$iteration" \
                                                    --top_k "$top_k" \
                                                    --train_size "$train_size" \
                                                    --batch_sizes "${batch_size_array[@]}" \
                                                    --monte_carlo_runs "$monte_carlo_runs" \
                                                    > "logs/eval_cross_${method}_${optimize_method}_${dataset}_${seed_prompt}_${with_rationale}_train${train_size}_iteration${iteration}_top${top_k}_bs${batch_sizes_tag}_mc${monte_carlo_runs}_eval_${EVAL_MODEL_TAG}_from_${OPT_MODEL_TAG}_trial${TRIAL}.log" 2>&1 &
                                                pids+=($!)
                                            done
                                        done

                                        for pid in "${pids[@]}"; do
                                            wait "$pid"
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
done
