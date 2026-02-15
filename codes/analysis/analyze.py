import os
import argparse
import matplotlib.pyplot as plt
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_values(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def plot_evolution(result_dir):
    if not os.path.exists(result_dir):
        print(f"Directory not found: {result_dir}")
        return

    # Check if data exists
    val_qwk_path = os.path.join(result_dir, 'val_qwk.txt')
    if not os.path.exists(val_qwk_path):
        return

    print(f"Analyzing results in: {result_dir}")

    # Load data
    val_qwk = load_values(val_qwk_path)
    val_acc = load_values(os.path.join(result_dir, 'val_accuracy.txt'))
    train_qwk = load_values(os.path.join(result_dir, 'train_qwk.txt'))
    train_acc = load_values(os.path.join(result_dir, 'train_accuracy.txt'))
    train_qwk_re = load_values(os.path.join(result_dir, 'train_qwk_re.txt'))
    
    # Extract info from path for title
    # Expected path ending: .../dataset/model_name/optimize_method/seed_prompt
    parts = result_dir.strip(os.sep).split(os.sep)
    if len(parts) >= 4:
        seed_prompt = parts[-1]
        optimize_method = parts[-2]
        model_name = parts[-3]
        dataset = parts[-4]
        # Check if trial info is in path
        trial = "0"
        for part in parts:
            if part.startswith("trial_"):
                trial = part
                break
            if part == "optimization_results":
                trial = "0"
                break
        
        title = f'Rubric Evolution: {dataset} - {model_name}\nMethod: {optimize_method}, Seed: {seed_prompt}, Trial: {trial}'
    else:
        title = f'Rubric Evolution: {result_dir}'

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Validation scores (recorded at step 0 and after each evolution)
    steps = range(len(val_qwk))
    plt.plot(steps, val_qwk, label='Validation QWK', marker='o', linewidth=2)
    plt.plot(steps, val_acc, label='Validation Accuracy', marker='x', linestyle='--', alpha=0.7)
    
    # Train scores (recorded during evolution loop, starting from step 1)
    if train_qwk:
        # train_qwk is recorded before evolution in each step
        train_steps = range(1, len(train_qwk) + 1)
        plt.plot(train_steps, train_qwk, label='Train QWK (Pre-evolution)', marker='s', alpha=0.5, linestyle=':')
    
    if train_qwk_re:
        # train_qwk_re is recorded after evolution in each step (using new rubric)
        train_re_steps = range(1, len(train_qwk_re) + 1)
        plt.plot(train_re_steps, train_qwk_re, label='Train QWK (Post-evolution)', marker='^', linestyle='-.')

    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1) 
    
    output_path = os.path.join(result_dir, 'evolution_plot.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

def find_and_plot_all(base_path):
    # Search in optimization_results (trial 0)
    opt_results = os.path.join(base_path, 'optimization_results')
    if os.path.exists(opt_results):
        for root, dirs, files in os.walk(opt_results):
            if 'val_qwk.txt' in files:
                plot_evolution(root)

    # Search in optimization_trials
    opt_trials = os.path.join(base_path, 'optimization_trials')
    if os.path.exists(opt_trials):
        for root, dirs, files in os.walk(opt_trials):
            if 'val_qwk.txt' in files:
                plot_evolution(root)

def main():
    # Check if any arguments were provided
    if len(sys.argv) == 1:
        print("No arguments provided. Generating plots for all results...")
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        find_and_plot_all(base_path)
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize_method', type=str, default='base', choices=['base', 'optimize_with_few_shot'])
    parser.add_argument('--trial', type=str, default='0')
    parser.add_argument('--seed_prompt', type=str, choices=['simplest', 'simple', 'expert'], default='expert')
    parser.add_argument('--model_name', type=str, default='openai/gpt-4.1')
    parser.add_argument('--dataset', type=str, default='asap_1')
    
    args = parser.parse_args()

    trial = args.trial
    model_name = args.model_name
    
    # Construct path relative to the root of the workspace
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if trial == '0':
        trials_dir = os.path.join(base_path, 'optimization_results')
    else:
        trials_dir = os.path.join(base_path, 'optimization_trials', f'trial_{trial}')

    result_model_name = model_name.replace('/', '_')
    result_dir = os.path.join(trials_dir, args.dataset, result_model_name, args.optimize_method, args.seed_prompt)

    plot_evolution(result_dir)

if __name__ == "__main__":
    main()
