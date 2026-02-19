from llm_router import get_llm_response_async, get_llm_responses_batch_async
import asyncio
import re
import os
from rubric_prompts import EVALUATION_PROMPT, ETS_RUBRIC, EXAMPLE_FORMAT, prepare_asap_rubric

from utils import prepare_ets_dataset, prepare_asap_dataset
from sklearn.metrics import cohen_kappa_score
import shutil
ASYNC_BATCH_SIZE = 100

PROMPT_FOR_EVOLUTION = """I provided an assistant with the following rubrics to perform an essay grading task for me:
```
{current_rubric}
```

The following are examples where the assistant's ratings did not match the desired ratings:
```
{examples}
```
Please analyze the rubrics and the examples, and then propose new rubrics that will help the assistant to perform better on this task.

Read all the assistant responses and reflect on the rationales given by the assistant. Identify any patterns or common themes in the rationales that led to incorrect ratings. Consider how the rubrics could be adjusted to better align with these patterns by providing clearer/detailed guidelines for the assistant to follow.

Provide the new rubrics within ``` blocks."""

def parse_rating(response):
    # Find the substring after 'Rating:' and then extract all numbers from it
    after = re.search(r'Rating:\s*(.*)', response)
    if not after:
        return None
    nums = re.findall(r'\d+', after.group(1))
    if nums:
        return int(nums[-1])  # Return the last matched number after 'Rating:'
    else:
        return None


def parse_rationale(response):
    match = re.search(r'Rationale:\s*\[?<*(.*?)>*\]?\s*Rating:', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def quality_to_rating(quality, dataset='ets'):
    if dataset == 'ets':
        mapping = {
            "high": [5, 4],
            "medium": [3],
            "low": [2, 1]
        }
        return mapping.get(quality, None)
    elif dataset.startswith('asap'):
        return quality  # direct mapping
    else:
        return None


def rating_to_quality(rating, dataset='ets'):
    if dataset == 'ets':
        if rating in [5, 4]:
            return "high"
        elif rating == 3:
            return "medium"
        elif rating in [2, 1]:
            return "low"
        else: 
            return None
    elif dataset.startswith('asap'):
        return rating  # direct mapping
    else:
        return None


def quality_to_desired_rating(quality, dataset='ets'):
    if dataset == 'ets':
        mapping = {
            "high": '4 or 5',
            "medium": '3',
            "low": '1 or 2'
        }
        return mapping.get(quality, None)
    elif dataset.startswith('asap'):
        return quality  # direct mapping
    else:
        return None


def scoring(llm_judged_response, quality, dataset='ets'):
    rating = quality_to_rating(quality, dataset)
    parsed_rating = parse_rating(llm_judged_response)
    parsed_rationale = parse_rationale(llm_judged_response)
    if parsed_rating is None:
        print(f"Could not parse rating from response: {llm_judged_response}")
        return None

    if rating is None:
        return None
    
    if dataset == 'ets':
        if parsed_rating in rating:
            return 1
        else:
            return 0
    elif dataset.startswith('asap'):
        if parsed_rating == rating:
            return 1
        else:
            return 0


def quality_to_score_for_qwk(quality, dataset='ets'):
    if dataset == 'ets':
        mapping = {
            "high": 3,
            "medium": 2,
            "low": 1
        }
        return mapping.get(quality, None)
    elif dataset.startswith('asap'):
        return quality  # direct mapping
    else:
        return None

def parse_new_rubric(response):
    response = response.replace('```markdown\n', '```\n')
    pattern = r'```(.*?)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        # Return the last fenced block match instead of the first
        all_matches = re.findall(pattern, response, re.DOTALL)
        if all_matches:
            return all_matches[-1].strip()
    else:
        # If no fenced block is found, return the last part starts with ```
        pattern = r'```(.*?)$' # *? means non-greedy matching
        match = re.search(pattern, response, re.DOTALL)
        if match:
            all_matches = re.findall(pattern, response, re.DOTALL)
            if all_matches:
                last_match = all_matches[-1]
                if len(last_match.strip()) < 100:  # if too short, ignore
                    return None
                return all_matches[-1].strip()
        else:
            return None


def calculate_accuracy(model_responses, batch, dataset='ets'):
    correct = 0
    total = 0
    for i, model_response in enumerate(model_responses):
        if not isinstance(model_response, str):
            print(f"Skipping invalid response at index {i}: {model_response}")
            continue
        data = batch[i]
        score = scoring(model_response, data["answer"], dataset)
        if score is not None:
            correct += score
            total += 1
    if total == 0:
        return 0.0
    return correct / total


def calculate_qwk_for_batch(model_responses, batch, dataset='ets'):
    y_true, y_pred = [], []
    valid_count = 0
    for i, model_response in enumerate(model_responses):
        if not isinstance(model_response, str):
            print(f"Skipping invalid response at index {i}: {model_response}")
            continue
        data = batch[i]
        true_score = quality_to_score_for_qwk(data["answer"], dataset)
        pred_quality = rating_to_quality(parse_rating(model_response), dataset)
        pred_score = quality_to_score_for_qwk(pred_quality, dataset)
        if true_score is not None and pred_score is not None:
            y_true.append(true_score)
            y_pred.append(pred_score)
            valid_count += 1
    if len(y_true) == 0:
        return -1.0
    print(f"Valid pairs for QWK calculation: {valid_count}/{len(batch)}")
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


async def evaluate_on_rubric(model_name, params, batch, rubric, evaluation_prompt, dataset):
    for data in batch:
        prompt_for_llm = evaluation_prompt.format(
            essay_prompt=data["essay_prompt"],
            response=data["response"],
            rubric=rubric
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


def make_examples_for_rubric_evolution(model_responses, batch, dataset='ets'):
    examples = []
    for i, model_response in enumerate(model_responses):
        data = batch[i]
        if not isinstance(model_response, str):
            print(f"Skipping invalid response at index {i}: {model_response}")
            continue
        # only failures
        if scoring(model_response, data["answer"], dataset) == 1:
            continue
        examples.append(EXAMPLE_FORMAT.format(
            essay_prompt=data["essay_prompt"],
            response=data["response"],
            rationale=parse_rationale(model_response),
            rating=parse_rating(model_response),
            desired_rating=quality_to_desired_rating(data["answer"], dataset)
        ))
    return examples

def prepare_dataset(dataset_name):
    evaluation_prompt = EVALUATION_PROMPT
    if dataset_name == 'ets':
        current_rubric = ETS_RUBRIC  # initial rubric
        train_data, val_data, test_data = prepare_ets_dataset()
        return evaluation_prompt, current_rubric, train_data, val_data, test_data
    elif dataset_name.startswith('asap_'):
        essay_set = int(dataset_name.split('_')[1])
        train_data, val_data, test_data = prepare_asap_dataset(essay_set)
        current_rubric = prepare_asap_rubric(essay_set)
        return evaluation_prompt, current_rubric, train_data, val_data, test_data
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")



async def main(args):
    evaluation_prompt, current_rubric, train_data, val_data, test_data = prepare_dataset(args.dataset)

    model_name = args.model_name
    params = eval(args.api_params)

    result_dir = './only_fail_results'
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
    _, val_accuracy, val_qwk = await evaluate_on_rubric(model_name, params, val_batch, current_rubric, evaluation_prompt, args.dataset)

    # save the current rubric
    with open(os.path.join(result_dir, 'initial_rubric.txt'), 'w') as f:
        f.write(current_rubric)
    # save the val score
    with open(os.path.join(result_dir, 'val_score.txt'), 'w') as f:
        f.write(f"Step 0, Validation Accuracy: {val_accuracy}\n")
        f.write(f"Step 0, Validation QWK: {val_qwk}\n")

    best_val_accuracy = val_accuracy
    best_val_qwk = val_qwk
    
    # Ensure we don't exceed available chunks
    for step in range(min(max_evolution_steps, len(batches))):
        # prepare input for each chunk
        batch = batches[step]
        # evaluate on training data
        train_responses, train_accuracy, train_qwk = await evaluate_on_rubric(model_name, params, batch, current_rubric, evaluation_prompt, args.dataset)
        print(f"Step {step+1}, Training Accuracy: {train_accuracy}")
        print(f"Step {step+1}, Training QWK: {train_qwk}")
        with open(os.path.join(result_dir, f'train_score.txt'), 'a') as f:
            f.write(f"Step {step+1}, Training Accuracy: {train_accuracy}\n")
            f.write(f"Step {step+1}, Training QWK: {train_qwk}\n")
        
        examples = make_examples_for_rubric_evolution(train_responses, batch, args.dataset)
        examples_str = '\n\n'.join(examples)
        prompt_for_evolution = PROMPT_FOR_EVOLUTION.format(
            current_rubric=current_rubric,
            examples=examples_str
        )
        if len(examples) == 0:
            print("No failures in this batch, skipping evolution step.")
            continue
        with open(os.path.join(result_dir, f'prompt_for_evolution_step_{step+1}.txt'), 'w') as f:
            f.write(prompt_for_evolution)
        evolution_response = await get_llm_response_async([{"role": "user", "content": prompt_for_evolution}], model_name, params)
        new_rubric = parse_new_rubric(evolution_response)
        if new_rubric is not None:
            print("New Rubric proposed:")
            # print(new_rubric)
        else:
            print("No new rubric proposed, next step.")
            print(evolution_response)
            print("#"*20)
            print("parsed rubric:")
            print(new_rubric)
            continue

        # Evaluate on validation data
        val_responses, val_accuracy, val_qwk = await evaluate_on_rubric(model_name, params, val_batch, new_rubric, evaluation_prompt, args.dataset)
        print(f"Step {step+1}, Validation Accuracy: {val_accuracy}")
        print(f"Step {step+1}, Validation QWK: {val_qwk}")
        with open(os.path.join(result_dir, f'rubric_step_{step+1}.txt'), 'w') as f:
            f.write(new_rubric if new_rubric is not None else current_rubric)
        with open(os.path.join(result_dir, 'val_score.txt'), 'a') as f:
            f.write(f"Step {step+1}, Validation Accuracy: {val_accuracy}\n")
            f.write(f"Step {step+1}, Validation QWK: {val_qwk}\n")
        if val_qwk > best_val_qwk:
            best_val_accuracy = val_accuracy
            best_val_qwk = val_qwk
            best_rubric = new_rubric
            current_rubric = new_rubric
            print("New best rubric found.")
            print("Best Validation QWK:", best_val_qwk)
            with open(os.path.join(result_dir, 'best_rubric.txt'), 'w') as f:
                f.write(best_rubric)
        else:
            print("No improvement on validation QWK, keeping current rubric.")
            current_rubric = current_rubric  # keep the current rubric
        
        # evaluation on batch
        model_responses, train_accuracy, train_qwk = await evaluate_on_rubric(model_name, params, batch, current_rubric, evaluation_prompt, args.dataset)
        print(f"Step {step+1}, Training Accuracy (with updated rubric): {train_accuracy}")
        print(f"Step {step+1}, Training QWK (with updated rubric): {train_qwk}")
        with open(os.path.join(result_dir, f'train_score.txt'), 'a') as f:
            f.write(f"Step {step+1}, Training Accuracy (with updated rubric): {train_accuracy}\n")
            f.write(f"Step {step+1}, Training QWK (with updated rubric): {train_qwk}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='openai/gpt-4.1')
    parser.add_argument('--dataset', type=str, default='asap_1', choices=['ets', 'asap_1', 'asap_2', 'asap_3', 'asap_4', 'asap_5', 'asap_6', 'asap_7', 'asap_8'])
    parser.add_argument('--api_params', type=str, default='{"temperature":0.8,"max_tokens":8192}')
    args = parser.parse_args()
    asyncio.run(main(args))





