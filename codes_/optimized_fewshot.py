from inference import parse_rating, quality_to_desired_rating, get_llm_response_async, parse_new_rubric, prepare_dataset, ASYNC_BATCH_SIZE, get_llm_responses_batch_async, calculate_accuracy, calculate_qwk_for_batch
import shutil
import asyncio
import re
import os

FEW_SHOT_EVALUATION_PROMPT = '''You are a rater for writing responses on a high-stakes English language exam for second language learners. You will be provided with a prompt and the test-taker's response. Your rating should be based on the rubric below, following the specified format.

Rubric:
{rubric}

Examples without rationales:
{examples}

Here is a new essay to rate.
# Essay Prompt
"""{essay_prompt}"""
# Response
"""{response}"""
# Output format:
Rating: [<<<Your rating here.>>>]'''

EXAMPLE_FORMAT = '''
Essay Prompt:
"""{essay_prompt}"""

Student's essay:
"""{response}"""

Rating: {rating}'''

def make_examples_for_few_shot_evolution(batch, dataset='ets'):
    examples = []
    for i, data in enumerate(batch):
        examples.append(EXAMPLE_FORMAT.format(
            essay_prompt=data["essay_prompt"],
            response=data["response"],
            rating=quality_to_desired_rating(data["answer"], dataset)
        ))
    return examples

async def evaluate_on_rubric_few_shot(model_name, params, batch, rubric, evaluation_prompt, examples, dataset):
    for data in batch:
        prompt_for_llm = evaluation_prompt.format(
            essay_prompt=data["essay_prompt"],
            response=data["response"],
            rubric=rubric,
            examples=examples
        )
        data['messages'] = [{"role": "user", "content": prompt_for_llm}]
    batch_responses = []
    for i in range(0, len(batch), ASYNC_BATCH_SIZE):
        sub_batch = batch[i:i + ASYNC_BATCH_SIZE]
        responses = await get_llm_responses_batch_async(
            [data['messages'] for data in sub_batch],
            model_name,
            params,
        )
        batch_responses.extend(responses)
    accuracy = calculate_accuracy(batch_responses, batch, dataset)
    qwk = calculate_qwk_for_batch(batch_responses, batch, dataset)
    return batch_responses, accuracy, qwk


async def main(args):
    evaluation_prompt, current_rubric, train_data, val_data, test_data = prepare_dataset(args.dataset)
    sanitized_model_name = args.model_name.replace('/', '_')
    best_rubric_path = f'./glh_results/{args.dataset}/{sanitized_model_name}/best_rubric.txt'
    best_rubric = open(best_rubric_path).read() if os.path.exists(best_rubric_path) else current_rubric
    current_rubric = best_rubric

    evaluation_prompt = FEW_SHOT_EVALUATION_PROMPT
    train_examples = make_examples_for_few_shot_evolution(train_data[:10], args.dataset)
    train_examples_str = "\n\n".join(train_examples)

    model_name = args.model_name
    params = eval(args.api_params)

    result_dir = f'./optimized_few_shot_results'
    os.makedirs(result_dir, exist_ok=True)
    result_dir = os.path.join(result_dir, args.dataset)
    os.makedirs(result_dir, exist_ok=True)
    result_model_name = model_name.replace('/', '_')
    result_dir = os.path.join(result_dir, result_model_name)
    #remove if exists
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)


    max_evolution_steps = 10
    max_batch_size = 10
    batches = [train_data[i:i + max_batch_size] for i in range(0, len(train_data), max_batch_size)]
    val_batch = val_data[:100]
    best_val_accuracy = 0.0
    best_val_qwk = -1.0
    best_rubric = current_rubric
    _, val_accuracy, val_qwk = await evaluate_on_rubric_few_shot(model_name, params, val_batch, current_rubric, evaluation_prompt, train_examples_str, args.dataset)

    # save the current rubric
    with open(os.path.join(result_dir, 'initial_rubric.txt'), 'w') as f:
        f.write(current_rubric)
    # save the val score
    with open(os.path.join(result_dir, 'val_score.txt'), 'w') as f:
        f.write(f"Step 0, Validation Accuracy: {val_accuracy}\n")
        f.write(f"Step 0, Validation QWK: {val_qwk}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='openai/gpt-4.1')
    parser.add_argument('--dataset', type=str, default='ets', choices=['ets', 'asap_1', 'asap_2', 'asap_3', 'asap_4', 'asap_5', 'asap_6', 'asap_7', 'asap_8'])
    parser.add_argument('--api_params', type=str, default='{"temperature":0.8,"max_tokens":8192}')
    args = parser.parse_args()
    asyncio.run(main(args))