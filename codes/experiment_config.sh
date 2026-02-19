#!/usr/bin/env bash

# 実験用シェルスクリプトの共通設定。
# 使い方:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/experiment_config.sh"
#   load_model_profile "inference_eval"
#   load_inference_eval_hparams
#   validate_model_params

load_model_profile() {
    local profile="$1"
    case "$profile" in
        inference_eval)
            MODEL_NAMES=(
                # "openai/gpt-4.1"
                # "google/gemini-2.5-flash"
                "openai/gpt-5-mini"
                # "google/gemini-2.5-pro"
                "qwen/qwen3-next-80b-a3b-instruct"
                "google/gemini-3-flash-preview"
            )
            PARAMS_LIST=(
                # '{"temperature":1.0,"max_tokens":8192,"provider":{"only":["OpenAI","Google"]}}'
                # '{"temperature":1.0,"reasoning":{"max_tokens": 0},"max_tokens":8192,"provider":{"only":["OpenAI","Google"]}}'
                '{"reasoning":{"effort":"low"},"max_tokens":8192,"provider":{"only":["OpenAI"]}}'
                # '{"temperature": 1.0,"top_p":0.95,"reasoning":{"max_tokens": 1024},"max_tokens":8192,"provider":{"only":["OpenAI","Google"]}}'
                '{"temperature":0.7,"top_p":0.8,"top_k":20,"max_tokens":8192,"provider":{"only":["Google"]}}'
                '{"reasoning":{"effort":"low"},"max_tokens":8192,"provider":{"only":["Google"]}}'
            )
            ;;
        legacy)
            MODEL_NAMES=(
                "openai/gpt-4.1"
                "openai/gpt-5-mini"
                "google/gemini-2.5-pro"
                # "claude-3-5-haiku-latest"
            )
            PARAMS_LIST=(
                '{"temperature":0.8,"max_tokens":8192,"provider":{"order":["OpenAI","Anthropic","Google AI Studio","Cerebras","Fireworks"],"require_parameters":True}}'
                '{"reasoning":{"effort":"low"},"max_tokens":8192,"provider":{"order":["OpenAI","Anthropic","Google AI Studio","Cerebras","Fireworks"],"require_parameters":True}}'
                '{"reasoning":{"max_tokens": 1024},"max_tokens":8192,"provider":{"order":["OpenAI","Anthropic","Google AI Studio","Cerebras","Fireworks"],"require_parameters":True}}'
            )
            ;;
        test_one_model)
            MODEL_NAMES=(
                "openai/gpt-4.1"
                # "google/gemini-2.5-flash"
                # "openai/gpt-5-mini"
                # "google/gemini-2.5-pro"
                # "qwen/qwen3-next-80b-a3b-instruct"
                # "meta-llama/llama-4-scout"
                # "meta-llama/llama-4-maverick"
            )
            PARAMS_LIST=(
                '{"temperature":1.0,"max_tokens":8192,"provider":{"only":["OpenAI","Google"]}}'
                # '{"temperature":1.0,"reasoning":{"max_tokens": 0},"max_tokens":8192,"provider":{"only":["OpenAI","Google"],"require_parameters":True}}'
                # '{"reasoning":{"effort":"low"},"max_tokens":8192,"provider":{"only":["OpenAI","Google"],"require_parameters":True}}'
                # '{"temperature": 1.0,"top_p":0.95,"reasoning":{"max_tokens": 1024},"max_tokens":8192,"provider":{"only":["OpenAI","Google"],"require_parameters":True}}'
                # '{"temperature":0.7,"top_p":0.8,"top_k":20,"max_tokens":8192,"provider":{"only":["OpenAI","Google"],"require_parameters":True}}'
                # '{"max_tokens":8192,"provider":{"only":["OpenAI","Google"],"require_parameters":True}}'
                # '{"max_tokens":8192,"provider":{"only":["OpenAI","Google"],"require_parameters":True}}'
            )
            ;;
        inference_plain)
            MODEL_NAMES=(
                "gpt-4.1"
                # "gpt-5-mini-2025-08-07"
                # "claude-3-5-haiku-latest"
                # "models/gemini-2.5-pro"
            )
            PARAMS_LIST=(
                "{}"
                # '{}'
                # '{"temperature":0.3}'
                # '{"temperature":0.3}'
            )
            ;;
        *)
            echo "Unknown model profile: $profile" >&2
            return 1
            ;;
    esac
}

load_inference_eval_hparams() {
    WITH_RATIONALES=(
        "True"
        # "False"
    )

    ITERATIONS=(
        "5"
        # "3"
        # "1"
    )

    TOP_KS=(
        "3"
    )

    TRAIN_SIZES=(
        # "4"
        # "8"
        # "20"
        # "50"
        "100"
    )

    BATCH_SIZE_SETS=(
        "4 8 12"
        # "2 4 8"
    )

    MONTE_CARLO_RUNS=(
        # "1"
        # "2"
        "4"
    )

    DATASETS=(
        "asap_1"
        # "ets"
        "ets3"
        "ASAP2"
    )

    SEED_PROMPTS=(
        "expert"
        # "simplest"
        # "simple"
    )

    TRIAL="0"
}

validate_model_params() {
    if [[ ${#MODEL_NAMES[@]} -ne ${#PARAMS_LIST[@]} ]]; then
        echo "MODEL_NAMES and PARAMS_LIST length mismatch: ${#MODEL_NAMES[@]} vs ${#PARAMS_LIST[@]}" >&2
        return 1
    fi
    return 0
}
