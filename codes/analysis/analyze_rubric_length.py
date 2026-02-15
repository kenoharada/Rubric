import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_rubric_length(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return len(f.read())

def collect_rubric_data(base_path):
    # Structure: data[dataset][optimize_method][seed_prompt][model_name] = [trial1_lengths, trial2_lengths, ...]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # Helper to process a root directory
    def process_root(root_dir):
        if not os.path.exists(root_dir):
            return

        # Walk through the directory
        for root, dirs, files in os.walk(root_dir):
            if 'initial_rubric.txt' in files:
                # Parse path
                # Expected: .../{dataset}/{model_name}/{optimize_method}/{seed_prompt}
                parts = root.strip(os.sep).split(os.sep)
                if len(parts) >= 4:
                    seed_prompt = parts[-1]
                    optimize_method = parts[-2]
                    model_name = parts[-3]
                    dataset = parts[-4]
                    
                    lengths = []
                    
                    # Step 0
                    l0 = get_rubric_length(os.path.join(root, 'initial_rubric.txt'))
                    if l0 is not None:
                        lengths.append(l0)
                    
                    # Subsequent steps
                    step = 1
                    while True:
                        fname = f'rubric_step_{step}.txt'
                        fpath = os.path.join(root, fname)
                        if os.path.exists(fpath):
                            l = get_rubric_length(fpath)
                            lengths.append(l)
                            step += 1
                        else:
                            break
                    
                    if lengths:
                        data[dataset][optimize_method][seed_prompt][model_name].append(lengths)

    # Process trial 0 (optimization_results)
    process_root(os.path.join(base_path, 'optimization_results'))

    # Process other trials (optimization_trials)
    trials_root = os.path.join(base_path, 'optimization_trials')
    if os.path.exists(trials_root):
        for trial_dir in os.listdir(trials_root):
            if trial_dir.startswith('trial_'):
                process_root(os.path.join(trials_root, trial_dir))
    
    return data

def plot_rubric_length_changes(data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dataset in data:
        for method in data[dataset]:
            for seed in data[dataset][method]:
                plt.figure(figsize=(12, 8))
                
                models = data[dataset][method][seed]
                has_data = False
                
                for model_name, trials in models.items():
                    if not trials:
                        continue
                    
                    # Determine min length to average safely
                    min_len = min(len(t) for t in trials)
                    if min_len == 0:
                        continue

                    # Truncate all trials to min_len
                    truncated_trials = [t[:min_len] for t in trials]
                    
                    # Calculate average lengths
                    arr = np.array(truncated_trials)
                    mean_len = np.mean(arr, axis=0)
                    std_len = np.std(arr, axis=0)
                    
                    steps = range(min_len)
                    
                    # Plot
                    line, = plt.plot(steps, mean_len, label=f'{model_name} (n={len(trials)})', marker='o')
                    plt.fill_between(steps, mean_len - std_len, mean_len + std_len, alpha=0.2, color=line.get_color())
                    has_data = True

                if has_data:
                    plt.title(f'Average Rubric Length Change\nDataset: {dataset}, Method: {method}, Seed: {seed}')
                    plt.xlabel('Iteration Step')
                    plt.ylabel('Character Count')
                    plt.legend()
                    plt.grid(True)
                    
                    filename = f'{dataset}_{method}_{seed}_rubric_length.png'
                    filepath = os.path.join(output_dir, filename)
                    plt.savefig(filepath)
                    print(f"Saved plot to {filepath}")
                
                plt.close()

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_path, 'analysis', 'rubric_length_plots')
    
    print("Collecting rubric data...")
    data = collect_rubric_data(base_path)
    
    print("Generating plots...")
    plot_rubric_length_changes(data, output_dir)

if __name__ == "__main__":
    main()
