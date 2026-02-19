"""
Script to analyze the evolution of rubrics during optimization.

Usage:
    python analysis/analyze_rubric.py --trial 1 --dataset asap_1 --model_name openai/gpt-4.1 --optimize_method base --seed_prompt expert

Arguments:
    --trial: Trial number (default: 1)
    --dataset: Dataset name (default: asap_1)
    --model_name: Model name (default: openai/gpt-4.1)
    --optimize_method: Optimization method (default: base)
    --seed_prompt: Seed prompt type (default: expert)
    --base_dir: Base directory for results (default: optimization_trials)
"""

import os
import argparse
import difflib
import sys
import json

# Add parent directory to path to import llm_router
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_router import get_llm_response

def read_file(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return f.read()

def analyze_diff(before_text: str, after_text: str, model_name: str = "openai/gpt-4o-2024-08-06"):
    """
    Analyzes the difference between two texts using structured output.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes changes between two texts."},
        {"role": "user", "content": f"Analyze the differences between the following two texts:\n\n--- BEFORE ---\n{before_text}\n\n--- AFTER ---\n{after_text}"}
    ]

    schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A concise summary of the changes."
            },
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "The category of the change (e.g., content, style, structure, correction)."
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of the specific change."
                        },
                        "significance": {
                            "type": "string",
                            "enum": ["minor", "major", "critical"],
                            "description": "The significance of the change."
                        }
                    },
                    "required": ["category", "description", "significance"],
                    "additionalProperties": False
                },
                "description": "A list of specific changes found."
            }
        },
        "required": ["summary", "changes"],
        "additionalProperties": False
    }

    config = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "diff_analysis",
                "strict": True,
                "schema": schema
            }
        }
    }

    response_content = get_llm_response(messages, model_name, config)
    
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON. Raw response: {response_content}")
        return None

def generate_html_report(history, result_dir, args):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Rubric Optimization Analysis - Trial {args.trial}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; line-height: 1.6; color: #333; background-color: #f4f4f9; }}
            h1 {{ margin: 0 0 20px 0; color: #2c3e50; font-size: 1.8em; }}
            .container {{ max-width: 1400px; margin: 20px auto; background: white; padding: 30px; box-shadow: 0 0 15px rgba(0,0,0,0.05); border-radius: 8px; }}
            
            .meta-info {{ margin-bottom: 30px; background: #f1f8ff; padding: 15px 20px; border-radius: 6px; border: 1px solid #c8e1ff; display: flex; gap: 20px; flex-wrap: wrap; }}
            .meta-item {{ font-size: 0.95em; }}
            .meta-label {{ font-weight: bold; color: #0366d6; }}

            .step {{ border: 1px solid #e1e4e8; margin-bottom: 40px; border-radius: 8px; overflow: hidden; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.03); }}
            .step-header {{ padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e1e4e8; background-color: #f6f8fa; }}
            .step-header.accepted {{ background-color: #f0fff4; border-bottom: 1px solid #c6f6d5; }}
            .step-header.rejected {{ background-color: #fff5f5; border-bottom: 1px solid #fed7d7; }}
            
            .step-title {{ font-size: 1.2em; font-weight: 600; color: #24292e; }}
            .step-metrics {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 0.95em; }}
            
            .status-badge {{ padding: 4px 10px; border-radius: 2em; font-weight: 600; font-size: 0.85em; margin-left: 15px; letter-spacing: 0.5px; }}
            .status-accepted {{ background-color: #28a745; color: white; }}
            .status-rejected {{ background-color: #d73a49; color: white; }}
            
            .analysis-box {{ padding: 20px; background-color: #fff; border-bottom: 1px solid #e1e4e8; }}
            .analysis-summary {{ font-style: italic; margin-bottom: 15px; color: #586069; border-left: 4px solid #0366d6; padding-left: 15px; background: #f8f9fa; padding: 10px 15px; border-radius: 0 4px 4px 0; }}
            
            .change-list {{ list-style: none; padding: 0; margin: 0; }}
            .change-item {{ margin-bottom: 8px; padding: 10px; background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px; font-size: 0.95em; display: flex; align-items: baseline; }}
            .change-tags {{ flex-shrink: 0; margin-right: 10px; }}
            .change-tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 700; color: white; margin-right: 5px; text-transform: uppercase; }}
            .tag-minor {{ background-color: #6a737d; }}
            .tag-major {{ background-color: #d29922; }}
            .tag-critical {{ background-color: #d73a49; }}
            .tag-category {{ background-color: #0366d6; }}
            .change-desc {{ color: #24292e; }}

            .diff-container {{ overflow-x: auto; padding: 0; background-color: white; }}
            
            /* Diff Table Styles */
            table.diff {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 12px; border-collapse: collapse; width: 100%; table-layout: fixed; }}
            table.diff td {{ padding: 4px 8px; white-space: pre-wrap; word-wrap: break-word; vertical-align: top; line-height: 1.4; color: #24292e; }}
            table.diff th {{ display: none; }}
            
            .diff_header {{ background-color: #f6f8fa; color: #babbbd; text-align: right; padding-right: 8px; width: 35px; border-right: 1px solid #e1e4e8; user-select: none; }}
            .diff_next {{ background-color: #c0c0c0; }}
            .diff_add {{ background-color: #e6ffec; color: #24292e; }}
            .diff_chg {{ background-color: #fffbe6; color: #24292e; }}
            .diff_sub {{ background-color: #ffebe9; color: #24292e; }}
            
            /* Hide legends */
            a[href^="#difflib_chg"] {{ display: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Rubric Optimization Analysis</h1>
            <div class="meta-info">
                <div class="meta-item"><span class="meta-label">Dataset:</span> {args.dataset}</div>
                <div class="meta-item"><span class="meta-label">Model:</span> {args.model_name}</div>
                <div class="meta-item"><span class="meta-label">Method:</span> {args.optimize_method}</div>
                <div class="meta-item"><span class="meta-label">Trial:</span> {args.trial}</div>
            </div>
    """

    for item in history:
        step = item['step']
        qwk = item['qwk']
        best_before = item['best_qwk_before']
        accepted = item['accepted']
        analysis = item['analysis']
        
        status_class = "accepted" if accepted else "rejected"
        status_text = "ACCEPTED" if accepted else "REJECTED"
        
        html_content += f"""
        <div class="step">
            <div class="step-header {status_class}">
                <div class="step-title">Step {step}</div>
                <div class="step-metrics">
                    QWK: <strong>{qwk:.4f}</strong> <span style="color: #586069; margin-left: 10px;">(Prev Best: {best_before:.4f})</span>
                    <span class="status-badge status-{status_class}">{status_text}</span>
                </div>
            </div>
        """
        
        if analysis:
            html_content += f"""
            <div class="analysis-box">
                <div class="analysis-summary"><strong>AI Summary:</strong> {analysis.get('summary', 'N/A')}</div>
                <ul class="change-list">
            """
            for change in analysis.get('changes', []):
                sig = change.get('significance', 'minor').lower()
                cat = change.get('category', 'general')
                desc = change.get('description', '')
                html_content += f"""
                    <li class="change-item">
                        <div class="change-tags">
                            <span class="change-tag tag-{sig}">{sig}</span>
                            <span class="change-tag tag-category">{cat}</span>
                        </div>
                        <div class="change-desc">{desc}</div>
                    </li>
                """
            html_content += "</ul></div>"
        else:
            html_content += '<div class="analysis-box" style="color: #666;">No analysis available.</div>'

        html_content += f"""
            <div class="diff-container">
                {item['diff_html']}
            </div>
        </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """
    
    output_path = os.path.join(result_dir, 'rubric_analysis.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=str, default='1')
    parser.add_argument('--dataset', type=str, default='asap_1')
    parser.add_argument('--model_name', type=str, default='openai/gpt-4.1')
    parser.add_argument('--optimize_method', type=str, default='base')
    parser.add_argument('--seed_prompt', type=str, default='expert')
    parser.add_argument('--base_dir', type=str, default='optimization_trials')
    
    args = parser.parse_args()

    model_dir_name = args.model_name.replace('/', '_')
    
    result_dir = os.path.join(
        args.base_dir,
        f"trial_{args.trial}",
        args.dataset,
        model_dir_name,
        args.optimize_method,
        args.seed_prompt
    )

    if not os.path.exists(result_dir):
        # Fallback for trial 0 or if user meant optimization_results
        if args.trial == '0':
             result_dir = os.path.join(
                'optimization_results',
                args.dataset,
                model_dir_name,
                args.optimize_method,
                args.seed_prompt
            )
    
    if not os.path.exists(result_dir):
        print(f"Directory not found: {result_dir}")
        return

    print(f"Analyzing results in: {result_dir}")

    initial_rubric_path = os.path.join(result_dir, 'initial_rubric.txt')
    initial_rubric = read_file(initial_rubric_path)
    
    if initial_rubric is None:
        print("initial_rubric.txt not found.")
        return

    val_qwk_path = os.path.join(result_dir, 'val_qwk.txt')
    val_qwks = []
    if os.path.exists(val_qwk_path):
        with open(val_qwk_path, 'r') as f:
            val_qwks = [float(line.strip()) for line in f if line.strip()]
    
    if not val_qwks:
        print("No QWK scores found.")
        return

    current_rubric = initial_rubric
    best_qwk = val_qwks[0]
    
    print(f"Initial QWK: {best_qwk}")

    history = []
    step = 1
    # val_qwks[0] is initial. val_qwks[1] is step 1.
    while step < len(val_qwks):
        rubric_path = os.path.join(result_dir, f'rubric_step_{step}.txt')
        proposed_rubric = read_file(rubric_path)
        
        if proposed_rubric is None:
            print(f"rubric_step_{step}.txt not found, stopping.")
            break
            
        qwk = val_qwks[step]
        
        print(f"\n{'='*20} Step {step} {'='*20}")
        print(f"QWK: {qwk} (Best so far: {best_qwk})")
        
        accepted = False
        if qwk > best_qwk:
            accepted = True
            print("Result: ACCEPTED (New Best)")
        else:
            print("Result: REJECTED")

        print(f"Diff (Current vs Proposed):")
        diff = difflib.unified_diff(
            current_rubric.splitlines(),
            proposed_rubric.splitlines(),
            fromfile=f'Current (Step {step-1} end)',
            tofile=f'Proposed (Step {step})',
            lineterm=''
        )
        diff_text = '\n'.join(list(diff))
        
        analysis = None
        if not diff_text:
            print("No changes in rubric text.")
        else:
            print(diff_text)
            print("\nAI Analysis of Changes:")
            analysis = analyze_diff(current_rubric, proposed_rubric)
            if analysis:
                print(f"Summary: {analysis.get('summary')}")
                print("Changes:")
                for change in analysis.get('changes', []):
                    print(f"  - [{change.get('significance', 'unknown').upper()}] {change.get('category')}: {change.get('description')}")
        
        # Generate HTML diff for report
        diff_html = difflib.HtmlDiff().make_table(
            current_rubric.splitlines(),
            proposed_rubric.splitlines(),
            fromdesc=f'Current (Step {step-1})',
            todesc=f'Proposed (Step {step})',
            context=True,
            numlines=3
        )

        history.append({
            "step": step,
            "qwk": qwk,
            "best_qwk_before": best_qwk,
            "accepted": accepted,
            "diff_html": diff_html,
            "analysis": analysis
        })

        if accepted:
            best_qwk = qwk
            current_rubric = proposed_rubric
            
        step += 1
    
    generate_html_report(history, result_dir, args)

if __name__ == "__main__":
    main()
