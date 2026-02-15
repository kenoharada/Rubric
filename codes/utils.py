import pandas as pd
import random
import numpy as np
from rubric_prompts import prepare_prompt_for_asap

def prepare_ets_dataset():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
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
    # shuffle train data
    random.shuffle(train_data)

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
    random.shuffle(test_data)
    return train_data, val_data, test_data[:100] # use 100 samples for test



def prepare_asap_dataset(essay_set=1):
    # fix random seed for reproducibility
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    train_data_path = "dataset/asap-aes/training_set_rel3.tsv"
    # val_data_path = "dataset/asap-aes/valid_set.tsv" # no labels
    # test_data_path = "dataset/asap-aes/test_set.tsv" # no labels

    data = pd.read_csv(train_data_path, sep="\t", encoding="latin-1")
    data = data[data["essay_set"] == essay_set]
    # shuffle the training data
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    data_for_experiment = []
    for _, row in data.iterrows():
        essay_prompt = prepare_prompt_for_asap(essay_set)
        response = row["essay"]
        answer = row["rater1_domain1"]
        another_answer = row["rater2_domain1"]
        data_for_experiment.append({
            "essay_prompt": essay_prompt,
            "response": response,
            "answer": int(answer),
            "human": int(another_answer)
        })

    train_val_test_split = [0.8, 0.1, 0.1]
    n = len(data_for_experiment)
    train_end = int(n * train_val_test_split[0])
    val_end = train_end + int(n * train_val_test_split[1])
    train_data = data_for_experiment[:train_end]
    val_data = data_for_experiment[train_end:val_end]
    test_data = data_for_experiment[val_end:]

    return train_data, val_data, test_data[:100]  # use 100 samples for test


def prepare_asap2_dataset(essay_set=1):
    essay_set_mapping = {
        1: 'Exploring Venus', 
        2: 'Facial action coding system',
        3: 'The Face on Mars',
        4: '"A Cowboy Who Rode the Waves"',
        5: 'Driverless cars',
        6: 'Does the electoral college work?',
        7: 'Car-free cities'
    }
    if essay_set != 0:
        essay_set_name = essay_set_mapping[essay_set]
    else:
        essay_set_name = None
    # fix random seed for reproducibility
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    data_path = "dataset/ASAP2/ASAP2_train_sourcetexts.csv"
    data = pd.read_csv(data_path)
    if essay_set_name is not None:
        data = data[data["prompt_name"] == essay_set_name]
    print(len(data))
    # shuffle the data
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    data_for_experiment = []
    for _, row in data.iterrows():
        data_for_experiment.append({
            "essay_prompt": row["assignment"],
            "response": row["full_text"],
            "answer": int(row["score"]),
            "prompt_name": row["prompt_name"],
            "source_text": row["source_text_1"]
        })

    train_val_test_split = [0.8, 0.1, 0.1]
    n = len(data_for_experiment)
    train_end = int(n * train_val_test_split[0])
    val_end = train_end + int(n * train_val_test_split[1])
    train_data = data_for_experiment[:train_end]
    val_data = data_for_experiment[train_end:val_end]
    test_data = data_for_experiment[val_end:]

    return train_data, val_data, test_data[:100]  # use 100 samples for test


if __name__ == "__main__":
    train_data, val_data, test_data = prepare_asap_dataset(essay_set=1)
    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Test data size: {len(test_data)}")
    from sklearn.metrics import cohen_kappa_score
    from inference import quality_to_score_for_qwk
    y_true = [quality_to_score_for_qwk(d["answer"], dataset='asap') for d in val_data]
    y_pred = [quality_to_score_for_qwk(d["human"], dataset='asap') for d in val_data]
    print("Cohen's kappa:", cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    # accuracy
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
    print("Accuracy:", accuracy)
