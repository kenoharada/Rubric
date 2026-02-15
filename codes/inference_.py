from llm_api_utils import get_llm_response_async
import asyncio
import re
import os



RUBRIC = '''## Score 5
An essay at this level largely accomplishes all of the following:
- effectively addresses the topic and task
- is well organized and well developed, using clearly appropriate explanations, exemplifications, and/or details
- displays unity, progression, and coherence
- displays consistent facility in the use of language, demonstrating syntactic variety, appropriate word choice, and idiomaticity, though it may have minor lexical or grammatical errors

## Score 4 
An essay at this level largely accomplishes all of the following:
- addresses the topic and task well, though some points may not be fully elaborated
- is generally well organized and well developed, using appropriate and sufficient explanations, exemplifications, and/or details
- displays unity, progression, and coherence, though it may contain occasional redundancy, digression, or unclear connections
- displays facility in the use of language, demonstrating syntactic variety and range of vocabulary, though it will probably have occasional noticeable minor errors in structure, word form, or use of idiomatic language that do not interfere with meaning

## Score 3
An essay at this level is marked by one or more of the following:
- addresses the topic and task using somewhat developed explanations, exemplifications, and/or details
- displays unity, progression, and coherence, though connection of ideas may be occasionally obscured
- may demonstrate inconsistent facility in sentence formation and word choice that may result in lack of clarity and occasionally obscure meaning
- may display accurate but limited range of syntactic structures and vocabulary

## Score 2
An essay at this level may reveal one or more of the following weaknesses:
- limited development in response to the topic and task
- inadequate organization or connection of ideas
- inappropriate or insufficient exemplifications, explanations, or details to support or illustrate generalizations in response to the task
- a noticeably inappropriate choice of words or word forms
- an accumulation of errors in sentence structure and/or usage

## Score 1
An essay at this level is seriously flawed by one or more of the following weaknesses:
- serious disorganization or underdevelopment
- little or no detail, or irrelevant specifics, or questionable responsiveness to the task
- serious and frequent errors in sentence structure or usage'''


SEED_PROMPT = '''You are a rater for writing responses on a high-stakes English language exam for second language learners. You will be provided with a prompt and the test-taker's response. Your rating should be based on the rubric below, following the specified format. There are rating samples of experts so that you can refer to those when rating.

# Essay Prompt
"""{essay_prompt}"""
# Response
"""{response}"""
# Rubric
"""{rubric}"""
# Output format:
Justification: [<<<Your justification here.>>>]
Rating: [<<<Your rating here.>>>]'''

PROMPT_FOR_EVOLUTION = """I provided an assistant with the following rubrics to perform an essay grading task for me:
```
{current_rubric}
```

The following are examples of different inputs to the assistant, the justifications for scores from the assistant, the scores from the assistant, and desired scores which I would like the assistant to achieve. 
```
{examples}
```
Please analyze the rubrics and the examples, and then propose new rubrics that will help the assistant to perform better on this task.

Read all the assistant responses and reflect on the justifications given by the assistant. Identify any patterns or common themes in the justifications that led to correct or incorrect ratings. Consider how the rubrics could be adjusted to better align with these patterns by providing clearer/detailed guidelines for the assistant to follow.

Provide the new rubrics within ``` blocks."""

EXAMPLE_FORMAT = '''Input for the assistant:
Essay Prompt:
"""{essay_prompt}"""
Essay to be rated:
"""{response}"""
Justification from the assistant:
"""{justification}"""
Score from the assistant:
"""{rating}"""
Desired score:
"""{desired_rating}"""'''


async def zero_shot(model_name, params, messages):
    response = await get_llm_response_async(model_name, params, messages)
    return response


async def process_chunk(model_name, params, chunk, semaphore=None):
    """
    Run a batch of LLM requests with optional concurrency limiting via semaphore.
    Each item in chunk must have a 'messages' key (list of chat messages).
    """
    async def run_one(messages):
        if semaphore:
            async with semaphore:
                return await zero_shot(model_name, params, messages)
        return await zero_shot(model_name, params, messages)

    tasks = [asyncio.create_task(run_one(item["messages"])) for item in chunk]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Propagate or stringify exceptions (keeps alignment with inputs)
    cleaned = []
    for r in results:
        if isinstance(r, Exception):
            cleaned.append(f"ERROR: {type(r).__name__}: {r}")
        else:
            cleaned.append(r)
    return cleaned


def prepare_dataset():
    import pandas as pd
    column_name = ["answer_id", "question_id", "nationality", "quality"]
    index_training_csv_path = "dataset/ETS/data/text/index-training.csv"
    index_training_df = pd.read_csv(index_training_csv_path, names=column_name)
    index_validation_csv_path = "dataset/ETS/data/text/index-dev.csv"
    index_validation_df = pd.read_csv(index_validation_csv_path, names=column_name)
    index_test_csv_path = "dataset/ETS/data/text/index-test.csv"
    test_column_name = ["answer_id", "question_id", "quality"]
    index_test_df = pd.read_csv(index_test_csv_path, names=test_column_name)

    prompt_path_template = "dataset/ETS/data/text/prompts/{question_id}.txt"
    response_path_template = "dataset/ETS/data/text/responses/original/{answer_id}"

    train_data = []
    for _, row in index_training_df.iterrows():
        question_id = row["question_id"]
        answer_id = row["answer_id"]
        quality = row["quality"]
        with open(prompt_path_template.format(question_id=question_id), "r") as f:
            essay_prompt = f.read()
        with open(response_path_template.format(answer_id=answer_id), "r") as f:
            response = f.read()
        train_data.append({
            "essay_prompt": essay_prompt,
            "response": response,
            "answer": quality
        })

    val_data = []
    for _, row in index_validation_df.iterrows():
        question_id = row["question_id"]
        answer_id = row["answer_id"]
        quality = row["quality"]
        with open(prompt_path_template.format(question_id=question_id), "r") as f:
            essay_prompt = f.read()
        with open(response_path_template.format(answer_id=answer_id), "r") as f:
            response = f.read()
        val_data.append({
            "essay_prompt": essay_prompt,
            "response": response,
            "answer": quality
        })


    test_data = []
    for _, row in index_test_df.iterrows():
        question_id = row["question_id"]
        answer_id = row["answer_id"]
        quality = row["quality"]
        with open(prompt_path_template.format(question_id=question_id), "r") as f:
            essay_prompt = f.read()
        with open(response_path_template.format(answer_id=answer_id), "r") as f:
            response = f.read()
        test_data.append({
            "essay_prompt": essay_prompt,
            "response": response,
            "answer": quality
        })
    return train_data, val_data, test_data


def parse_rating(response):
    match = re.search(r'Rating:\s*\[?(\d+)\]?', response)
    if match:
        return int(match.group(1))
    else:
        return None


def parse_justification(response):
    match = re.search(r'Justification:\s*\[?(.*?)\]?\s*Rating:', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def quality_to_label(quality):
    mapping = {
        "high": [5, 4],
        "medium": [3],
        "low": [2, 1]
    }
    return mapping.get(quality, None)


def label_to_quality(label):
    if label in [5, 4]:
        return "high"
    elif label == 3:
        return "medium"
    elif label in [2, 1]:
        return "low"
    else:
        return None


def scoring(response, label):
    label = quality_to_label(label)
    parsed_rating = parse_rating(response)
    parsed_justification = parse_justification(response)
    # print('Justification:', parsed_justification)
    # print('Rating:', parsed_rating)
    if parsed_rating is None:
        return None
    
    if label is None:
        return None
    if parsed_rating in label:
        return 1
    else:
        return 0


def calculate_qwk(y_true, y_pred):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def parse_new_rubric(response):
    match = re.search(r'```(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def quality_to_desired_rating(quality):
    mapping = {
        "high": '4 or 5',
        "medium": '3',
        "low": '1 or 2'
    }
    return mapping.get(quality, None)


def qulality_to_score_for_qwk(quality):
    mapping = {
        "high": 3,
        "medium": 2,
        "low": 1
    }
    return mapping.get(quality, None)


def prepare_prompt(essay_prompt, response, rubric):
    prompt_for_llm = SEED_PROMPT.format(
        essay_prompt=essay_prompt,
        response=response,
        rubric=rubric
    )
    return prompt_for_llm


async def main(args):
    current_rubric = RUBRIC  # initial rubric
    train_data, val_data, test_data = prepare_dataset()
    model_name = args.model_name
    params = eval(args.api_params)

    result_dir = './results'
    os.makedirs(result_dir, exist_ok=True)
    result_model_name = model_name.replace('/', '_')
    result_dir = os.path.join(result_dir, result_model_name)
    #remove if exists
    if os.path.exists(result_dir):
        import shutil
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)


    max_evolution_steps = 10
    max_chunk_size = 10
    chunks = [train_data[i:i + max_chunk_size] for i in range(0, len(train_data), max_chunk_size)]
    val_chunk = val_data[:100]
    best_val_accuracy = 0.0
    best_val_qwk = -1.0
    best_rubric = current_rubric
    semaphore = asyncio.Semaphore(20)  # 同時実行数を20に制限
    val_results = await process_chunk(model_name, params, [{"messages": [{"role": "user", "content": prepare_prompt(data["essay_prompt"], data["response"], current_rubric)}]} for data in val_chunk], semaphore)
    val_histories = []
    for i, model_response in enumerate(val_results):
        data = val_chunk[i]
        score = scoring(model_response, data["answer"])
        val_histories.append({
            "essay_prompt": data["essay_prompt"],
            "response": data["response"],
            "justification": parse_justification(model_response),
            "rating": parse_rating(model_response),
            "desired_rating": quality_to_desired_rating(data["answer"]),
            "score": score,
            "label": data["answer"]
        })
    val_accuracy = sum([h['score'] for h in val_histories if h['score'] is not None]) / len(val_histories)
    print(f"Step 0, Validation Accuracy: {val_accuracy}")
    y_true, y_pred = [], []
    for h in val_histories:
        true_score = qulality_to_score_for_qwk(h["label"])
        pred_quality = label_to_quality(h["rating"])
        pred_score = qulality_to_score_for_qwk(pred_quality)
        if true_score is not None and pred_score is not None:
            y_true.append(true_score)
            y_pred.append(pred_score)
    val_qwk = calculate_qwk(y_true, y_pred)
    print(f"Step 0, Validation QWK: {val_qwk}")
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
    for step in range(min(max_evolution_steps, len(chunks))):
        # prepare input for each chunk
        input_chunk = chunks[step]
        histories = []
        for data in input_chunk:
            prompt_for_llm = prepare_prompt(data["essay_prompt"], data["response"], current_rubric)
            data['prompt_for_autograde'] = prompt_for_llm
            data['messages'] = [{"role": "user", "content": prompt_for_llm}]
        # process chunks
        results = await process_chunk(model_name, params, input_chunk)

        # evaluate results
        for i, model_response in enumerate(results):
            data = input_chunk[i]
            score = scoring(model_response, data["answer"])
            histories.append({
                "essay_prompt": data["essay_prompt"],
                "response": data["response"],
                "justification": parse_justification(model_response),
                "rating": parse_rating(model_response),
                "desired_rating": quality_to_desired_rating(data["answer"]),
                "score": score,
                "label": data["answer"]
            })
        # Accuracy
        accuracy = sum([h['score'] for h in histories if h['score'] is not None]) / len(histories)
        print(f"Step {step+1}, Training Accuracy: {accuracy}")
        # QWK
        y_true, y_pred = [], []
        for h in histories:
            true_score = qulality_to_score_for_qwk(h["label"])
            pred_quality = label_to_quality(h["rating"])
            pred_score = qulality_to_score_for_qwk(pred_quality)
            if true_score is not None and pred_score is not None:
                y_true.append(true_score)
                y_pred.append(pred_score)
        qwk = calculate_qwk(y_true, y_pred)
        
        print(f"Step {step+1}, Training QWK: {qwk}")
        with open(os.path.join(result_dir, f'train_score.txt'), 'a') as f:
            f.write(f"Step {step+1}, Training Accuracy: {accuracy}\n")
            f.write(f"Step {step+1}, Training QWK: {qwk}\n")
        # Validate on validation data
        val_histories = []

        examples = []
        for h in histories:
            examples.append(EXAMPLE_FORMAT.format(
                essay_prompt=h["essay_prompt"],
                response=h["response"],
                justification=h["justification"],
                rating=h["rating"],
                desired_rating=h["desired_rating"],
            ))
        examples_str = '\n\n'.join(examples)
        prompt_for_evolution = PROMPT_FOR_EVOLUTION.format(
            current_rubric=current_rubric,
            examples=examples_str
        )
        evolution_response = await zero_shot(model_name, params, [{"role": "user", "content": prompt_for_evolution}])
        new_rubric = parse_new_rubric(evolution_response)
        if new_rubric is not None:
            print("New Rubric:")
            print(new_rubric)
        else:
            print("No new rubric proposed, next step.")
            print(evolution_response)
            continue
        
        # Evaluate on validation data
        val_results = await process_chunk(model_name, params, [{"messages": [{"role": "user", "content": prepare_prompt(data["essay_prompt"], data["response"], new_rubric)}]} for data in val_chunk])
        val_histories = []
        for i, model_response in enumerate(val_results):
            data = val_chunk[i]
            score = scoring(model_response, data["answer"])
            val_histories.append({
                "essay_prompt": data["essay_prompt"],
                "response": data["response"],
                "justification": parse_justification(model_response),
                "rating": parse_rating(model_response),
                "desired_rating": quality_to_desired_rating(data["answer"]),
                "score": score,
                "label": data["answer"]
            })
        val_accuracy = sum([h['score'] for h in val_histories if h['score'] is not None]) / len(val_histories)
        print(f"Step {step+1}, Validation Accuracy: {val_accuracy}")
        y_true, y_pred = [], []
        for h in val_histories:
            true_score = qulality_to_score_for_qwk(h["label"])
            pred_quality = label_to_quality(h["rating"])
            pred_score = qulality_to_score_for_qwk(pred_quality)
            if true_score is not None and pred_score is not None:
                y_true.append(true_score)
                y_pred.append(pred_score)
        val_qwk = calculate_qwk(y_true, y_pred)
        print(f"Step {step+1}, Validation QWK: {val_qwk}")
        with open(os.path.join(result_dir, f'rubric_step_{step+1}.txt'), 'w') as f:
            f.write(new_rubric if new_rubric is not None else current_rubric)
        with open(os.path.join(result_dir, 'val_score.txt'), 'a') as f:
            f.write(f"Step {step+1}, Validation Accuracy: {val_accuracy}\n")
            f.write(f"Step {step+1}, Validation QWK: {val_qwk}\n")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_rubric = new_rubric
            current_rubric = new_rubric
            print("New best rubric found.")
            with open(os.path.join(result_dir, 'best_rubric.txt'), 'w') as f:
                f.write(best_rubric)
        
        # Evaluate on train data
        if new_rubric is None:
            continue
        for data in input_chunk:
            prompt_for_llm = prepare_prompt(data["essay_prompt"], data["response"], current_rubric)
            data['prompt_for_autograde'] = prompt_for_llm
            data['messages'] = [{"role": "user", "content": prompt_for_llm}]
        # process chunks
        results = await process_chunk(model_name, params, input_chunk)
        histories = []
        for i, model_response in enumerate(results):
            data = input_chunk[i]
            score = scoring(model_response, data["answer"])
            histories.append({
                "essay_prompt": data["essay_prompt"],
                "response": data["response"],
                "justification": parse_justification(model_response),
                "rating": parse_rating(model_response),
                "desired_rating": quality_to_desired_rating(data["answer"]),
                "score": score,
                "label": data["answer"]
            })
        accuracy = sum([h['score'] for h in histories if h['score'] is not None]) / len(histories)
        print(f"Step {step+1}, Training Accuracy (re-evaluated): {accuracy}")
        y_true, y_pred = [], []
        for h in histories:
            true_score = qulality_to_score_for_qwk(h["label"])
            pred_quality = label_to_quality(h["rating"])
            pred_score = qulality_to_score_for_qwk(pred_quality)
            if true_score is not None and pred_score is not None:
                y_true.append(true_score)
                y_pred.append(pred_score)
        qwk = calculate_qwk(y_true, y_pred)
        print(f"Step {step+1}, Training QWK (re-evaluated): {qwk}")
        with open(os.path.join(result_dir, f'train_score.txt'), 'a') as f:
            f.write(f"Step {step+1}, Training Accuracy (re-evaluated): {accuracy}\n")
            f.write(f"Step {step+1}, Training QWK (re-evaluated): {qwk}\n")
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-4.1')
    parser.add_argument('--api_params', type=str, default='{}')
    args = parser.parse_args()
    asyncio.run(main(args))





