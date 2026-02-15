import ast
import asyncio
import json
import os
import shutil

from inference import (
    ASYNC_BATCH_SIZE,
    calculate_accuracy,
    calculate_qwk_for_batch,
    evaluate_on_rubric,
    get_llm_responses_batch_async,
    prepare_dataset,
    quality_to_desired_rating,
)


FEW_SHOT_EVALUATION_PROMPT = """You are a rater for writing responses on a high-stakes English language exam for second language learners. You will be provided with a prompt and the test-taker's response. Your rating should be based on the rubric below, following the specified format.

Rubric:
{rubric}

Examples:
{examples}

Here is a new essay to rate.
# Essay Prompt
\"\"\"{essay_prompt}\"\"\"
# Response
\"\"\"{response}\"\"\"
# Output format:
Rating: [<<<Your rating here.>>>]"""

FEW_SHOT_EXAMPLE_FORMAT = """
Essay Prompt:
\"\"\"{essay_prompt}\"\"\"

Student's essay:
\"\"\"{response}\"\"\"

Rating: {rating}"""


def make_examples_for_few_shot_evolution(batch, dataset="ets"):
    examples = []
    for data in batch:
        examples.append(
            FEW_SHOT_EXAMPLE_FORMAT.format(
                essay_prompt=data["essay_prompt"],
                response=data["response"],
                rating=quality_to_desired_rating(data["answer"], dataset),
            )
        )
    return examples


def build_inference_run_name(args, include_sampling=True):
    base_name = (
        f"{args.optimize_method.split('_')[0]}_{args.seed_prompt}_{args.with_rationale}"
        f"_train{args.train_size}_iteration{args.iteration}_top{args.top_k}"
    )
    if not include_sampling:
        return base_name
    batch_sizes_tag = "-".join(map(str, args.batch_sizes))
    return f"{base_name}_bs{batch_sizes_tag}_mc{args.monte_carlo_runs}"


async def evaluate_on_rubric_few_shot(
    model_name, params, batch, rubric, evaluation_prompt, examples, dataset
):
    for data in batch:
        prompt_for_llm = evaluation_prompt.format(
            essay_prompt=data["essay_prompt"],
            response=data["response"],
            rubric=rubric,
            examples=examples,
        )
        data["messages"] = [{"role": "user", "content": prompt_for_llm}]

    batch_responses = []
    for i in range(0, len(batch), ASYNC_BATCH_SIZE):
        sub_batch = batch[i : i + ASYNC_BATCH_SIZE]
        responses = await get_llm_responses_batch_async(
            [data["messages"] for data in sub_batch],
            model_name,
            params,
        )
        batch_responses.extend(responses)

    accuracy = calculate_accuracy(batch_responses, batch, dataset)
    qwk = calculate_qwk_for_batch(batch_responses, batch, dataset)
    return batch_responses, accuracy, qwk


def parse_api_params(params_text):
    try:
        parsed = json.loads(params_text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(params_text)

    if not isinstance(parsed, dict):
        raise ValueError(f"api_params must be a dict-like string, got: {type(parsed)}")
    return parsed


def return_inference_rubric_path(args):
    if args.trial == "0":
        result_dir = "./optimization_results"
    else:
        result_dir = f"./optimization_trials/trial_{args.trial}"

    run_name_candidates = [
        build_inference_run_name(args, include_sampling=True),
        build_inference_run_name(args, include_sampling=False),
    ]
    opt_model_dir = args.opt_model_name.replace("/", "_")

    for run_name in run_name_candidates:
        candidate_dir = os.path.join(result_dir, args.dataset, opt_model_dir, run_name)
        best_rubric_path = os.path.join(candidate_dir, "best_rubric.txt")
        if os.path.exists(best_rubric_path):
            return best_rubric_path, run_name

    raise FileNotFoundError(
        "best_rubric.txt not found for optimization model settings. "
        f"dataset={args.dataset}, model={args.opt_model_name}, "
        f"run_names={run_name_candidates}, trial={args.trial}"
    )


def build_cross_eval_run_name(args, inference_run_name):
    if args.optimize_method == "no_optimize":
        return (
            f"{args.method}_no_optimize_{args.seed_prompt}"
            f"_from_{args.opt_model_name.replace('/', '_')}"
        )
    return (
        f"{args.method}_{inference_run_name}"
        f"_from_{args.opt_model_name.replace('/', '_')}"
    )


async def main(args):
    evaluation_prompt, current_rubric, train_data, _, test_data = prepare_dataset(
        args.dataset, args.seed_prompt
    )

    rubric_source_path = None
    if args.optimize_method != "no_optimize":
        rubric_source_path, inference_run_name = return_inference_rubric_path(args)
        with open(rubric_source_path, "r") as f:
            current_rubric = f.read()
    else:
        inference_run_name = f"{args.optimize_method}_{args.seed_prompt}"

    if "few_shot" in args.method:
        evaluation_prompt = FEW_SHOT_EVALUATION_PROMPT
        train_examples = make_examples_for_few_shot_evolution(
            train_data[: args.few_shot_num], args.dataset
        )
        examples = "\n".join(train_examples)
    else:
        examples = ""

    eval_model_name = args.eval_model_name
    eval_params = parse_api_params(args.eval_api_params)

    if args.trial == "0":
        result_root = "./evaluation_results_cross_model"
    else:
        os.makedirs("./evaluation_cross_trials", exist_ok=True)
        result_root = f"./evaluation_cross_trials/trial_{args.trial}"

    os.makedirs(result_root, exist_ok=True)
    result_dir = os.path.join(
        result_root, args.dataset, eval_model_name.replace("/", "_")
    )
    os.makedirs(result_dir, exist_ok=True)

    run_name = build_cross_eval_run_name(args, inference_run_name)
    result_dir = os.path.join(result_dir, run_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "rubric.txt"), "w") as f:
        f.write(current_rubric)

    metadata = {
        "dataset": args.dataset,
        "method": args.method,
        "optimize_method": args.optimize_method,
        "seed_prompt": args.seed_prompt,
        "with_rationale": args.with_rationale,
        "iteration": args.iteration,
        "top_k": args.top_k,
        "train_size": args.train_size,
        "batch_sizes": args.batch_sizes,
        "monte_carlo_runs": args.monte_carlo_runs,
        "trial": args.trial,
        "eval_model_name": args.eval_model_name,
        "eval_api_params": args.eval_api_params,
        "opt_model_name": args.opt_model_name,
        "opt_api_params": args.opt_api_params,
        "rubric_source_path": rubric_source_path,
    }
    with open(os.path.join(result_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    eval_params["user"] = (
        f"eval_cross_{args.dataset}_{args.eval_model_name.replace('/', '_')}"
        f"_from_{args.opt_model_name.replace('/', '_')}_{run_name}"
    )

    if "few_shot" in args.method:
        model_responses, test_accuracy, test_qwk = await evaluate_on_rubric_few_shot(
            eval_model_name,
            eval_params,
            test_data,
            current_rubric,
            evaluation_prompt,
            examples,
            args.dataset,
        )
    else:
        model_responses, test_accuracy, test_qwk = await evaluate_on_rubric(
            eval_model_name,
            eval_params,
            test_data,
            current_rubric,
            evaluation_prompt,
            args.dataset,
        )

    results = []
    for model_response, data in zip(model_responses, test_data):
        results.append(
            {
                "essay_prompt": data["essay_prompt"],
                "essay_response": data["response"],
                "annotated_score": data["answer"],
                "model_response": model_response,
            }
        )

    with open(os.path.join(result_dir, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(result_dir, "accuracy.txt"), "w") as f:
        f.write(str(test_accuracy))
    with open(os.path.join(result_dir, "qwk.txt"), "w") as f:
        f.write(str(test_qwk))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    def positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"Expected a positive integer, got: {value}")
        return ivalue

    parser.add_argument(
        "--method", type=str, default="zero_shot", choices=["zero_shot", "few_shot"]
    )
    parser.add_argument(
        "--optimize_method",
        type=str,
        default="base",
        choices=["base", "optimize_with_few_shot", "no_optimize"],
    )
    parser.add_argument("--trial", type=str, default="0")
    parser.add_argument(
        "--seed_prompt",
        type=str,
        choices=["simplest", "simple", "expert", "self"],
        default="expert",
    )
    parser.add_argument("--dataset", type=str, default="asap_1", choices=["ets", "ets3", "asap_1", "asap_2", "asap_3", "asap_4", "asap_5", "asap_6", "asap_7", "asap_8", "ASAP2"])
    parser.add_argument("--few_shot_num", type=int, default=4)

    parser.add_argument("--eval_model_name", type=str, default="openai/gpt-4.1")
    parser.add_argument(
        "--eval_api_params", type=str, default='{"temperature":0.8,"max_tokens":8192}'
    )
    parser.add_argument("--opt_model_name", type=str, default="openai/gpt-4.1")
    parser.add_argument(
        "--opt_api_params", type=str, default='{"temperature":0.8,"max_tokens":8192}'
    )

    parser.add_argument("--with_rationale", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--iteration", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--train_size", type=int, default=20)
    parser.add_argument("--batch_sizes", type=positive_int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--monte_carlo_runs", type=positive_int, default=1)

    args = parser.parse_args()
    asyncio.run(main(args))
