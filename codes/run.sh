bash inference.sh
bash evaluate.sh

# MODEL_NAME="openai/gpt-5-mini"
# PARAMS='{"reasoning":{"effort":"low"},"max_tokens":8192,"provider":{"only":["OpenAI","Google"]}}'
# OPTIMIZE_METHOD="base"
# METHODS=(
#     "zero_shot"
#     "few_shot"
# )
# WITH_RATIONALES=(
#     "True"
#     "False"
# )
# ITERATIONS=(
#     "5"
#     "1"
# )
# TOP_K="3"
# TRAIN_SIZE="100"
# TRIAL="0"
# SEED_PROMPT="expert"
# DATASETS=(
#     "asap_1"
#     "ets"
# )
# for dataset in "${DATASETS[@]}"; do
#     for method in "${METHODS[@]}"; do
#         for with_rationale in "${WITH_RATIONALES[@]}"; do
#             for iteration in "${ITERATIONS[@]}"; do
#                 if [[ "$with_rationale" == "True" && "$iteration" == "1" ]]; then
#                     continue
#                 fi
#                 if [[ "$with_rationale" == "False" && "$iteration" != "1" ]]; then
#                     continue
#                 fi
#                 python3 evaluate.py \
#                     --model_name "$MODEL_NAME" \
#                     --api_params "$PARAMS" \
#                     --dataset "$dataset" \
#                     --method "$method" \
#                     --optimize_method "$OPTIMIZE_METHOD" \
#                     --seed_prompt "$SEED_PROMPT" \
#                     --trial "$TRIAL" \
#                     --with_rationale "$with_rationale" \
#                     --iteration "$iteration" \
#                     --top_k "$TOP_K" \
#                     --train_size "$TRAIN_SIZE"
#             done
#         done
#     done
# done
