import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_values(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def collect_data(base_path):
    # Structure: data[dataset][optimize_method][seed_prompt][model_name] = [trial1_scores, trial2_scores, ...]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # Helper to process a root directory
    def process_root(root_dir):
        if not os.path.exists(root_dir):
            return

        # Walk through the directory
        for root, dirs, files in os.walk(root_dir):
            if 'val_qwk.txt' in files:
                # Parse path
                # Expected: .../{dataset}/{model_name}/{optimize_method}/{seed_prompt}
                parts = root.strip(os.sep).split(os.sep)
                if len(parts) >= 4:
                    seed_prompt = parts[-1]
                    optimize_method = parts[-2]
                    model_name = parts[-3]
                    dataset = parts[-4]
                    
                    # Load scores
                    scores = load_values(os.path.join(root, 'val_qwk.txt'))
                    if scores:
                        data[dataset][optimize_method][seed_prompt][model_name].append(scores)

    # Process trial 0 (optimization_results)
    process_root(os.path.join(base_path, 'optimization_results'))

    # Process other trials (optimization_trials)
    trials_root = os.path.join(base_path, 'optimization_trials')
    if os.path.exists(trials_root):
        for trial_dir in os.listdir(trials_root):
            if trial_dir.startswith('trial_'):
                process_root(os.path.join(trials_root, trial_dir))
    
    return data

def plot_averaged_relative_changes(data, output_dir):
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
                    
                    # Calculate relative changes for each trial
                    # relative_change[i] = score[i] - score[0]
                    relative_changes = []
                    for t in truncated_trials:
                        start_score = t[0]
                        rel = [x - start_score for x in t]
                        relative_changes.append(rel)
                    
                    # Convert to numpy array for easy averaging
                    arr = np.array(relative_changes)
                    mean_change = np.mean(arr, axis=0)
                    std_change = np.std(arr, axis=0)
                    
                    steps = range(min_len)
                    
                    # Plot
                    line, = plt.plot(steps, mean_change, label=f'{model_name} (n={len(trials)})', marker='o')
                    plt.fill_between(steps, mean_change - std_change, mean_change + std_change, alpha=0.2, color=line.get_color())
                    has_data = True

                if has_data:
                    plt.title(f'Average Relative QWK Improvement\nDataset: {dataset}, Method: {method}, Seed: {seed}')
                    plt.xlabel('Iteration Step')
                    plt.ylabel('Relative QWK Change (vs Step 0)')
                    plt.legend()
                    plt.grid(True)
                    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    
                    filename = f'{dataset}_{method}_{seed}_relative_comparison.png'
                    filepath = os.path.join(output_dir, filename)
                    plt.savefig(filepath)
                    print(f"Saved plot to {filepath}")
                
                plt.close()

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_path, 'analysis', 'comparison_plots')
    
    print("Collecting data...")
    data = collect_data(base_path)
    
    print("Generating plots...")
    plot_averaged_relative_changes(data, output_dir)

if __name__ == "__main__":
    main()
