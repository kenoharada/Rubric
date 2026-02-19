from inference import parse_rating, quality_to_desired_rating, evaluate_on_rubric, get_llm_response_async, parse_new_rubric, prepare_dataset
import shutil
import asyncio
import re
import os


GLH_EVOLUTION_PROMPT = """You are a professional teacher of English language for second language learners. For evaluting students' essay, you are creating grading rubric. You will be given a initial grading rubric, and several examples of correctly grading to some students' essays. 
The initial grading rubric is: 
{current_rubric}

The samples of correctly grading are:
{examples} 

Your generated grading criteria should illustrate the key points of the answer for each question or sub-question and the rules for assigning points. Let's think step by step. Once you made a final analyisis result, provide the new rubrics within ``` blocks."""

EXAMPLE_FORMAT = '''
Essay Prompt:
"""{essay_prompt}"""
Student's essay:
"""{response}"""
Desired score:
"""{desired_rating}"""'''

def make_examples_for_glh_rubric_evolution(model_responses, batch, dataset='ets'):
    examples = []
    for i, model_response in enumerate(model_responses):
        data = batch[i]
        if not isinstance(model_response, str):
            print(f"Skipping invalid response at index {i}: {model_response}")
            continue
        examples.append(EXAMPLE_FORMAT.format(
            essay_prompt=data["essay_prompt"],
            response=data["response"],
            desired_rating=quality_to_desired_rating(data["answer"], dataset)
        ))
    return examples


async def main(args):
    evaluation_prompt, current_rubric, train_data, val_data, test_data = prepare_dataset(args.dataset)

    model_name = args.model_name
    params = eval(args.api_params)

    result_dir = f'./{args.method}_results'
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
        
        examples = make_examples_for_glh_rubric_evolution(train_responses, batch, args.dataset)
        examples_str = '\n\n'.join(examples)
        prompt_for_evolution = GLH_EVOLUTION_PROMPT.format(
            current_rubric=current_rubric,
            examples=examples_str
        )
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
    parser.add_argument('--method', type=str, default='glh', choices=['glh', 'standard'])
    parser.add_argument('--model_name', type=str, default='openai/gpt-4.1')
    parser.add_argument('--dataset', type=str, default='ets', choices=['ets', 'asap_1', 'asap_2', 'asap_3', 'asap_4', 'asap_5', 'asap_6', 'asap_7', 'asap_8'])
    parser.add_argument('--api_params', type=str, default='{"temperature":0.8,"max_tokens":8192}')
    args = parser.parse_args()
    asyncio.run(main(args))