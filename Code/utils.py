import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_experiment_run(run_dir):
    """
    Analyze and report results from an experiment run directory.
    
    Args:
        run_dir (str): Path to the experiment run directory
    
    Returns:
        dict: Dictionary containing analysis results
    """
    results = {}
    
    # Load model configuration
    config_path = os.path.join(run_dir, 'models/configs', 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)
    
    # Load model performance metrics
    metrics_path = os.path.join(run_dir, 'metrics/summary', 'evaluation_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)
    
    # Load training history
    history_path = os.path.join(run_dir, 'metrics/training_history', 'history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            results['history'] = json.load(f)
    
    # Generate summary report
    summary = {
        'Experiment Details': {
            'Timestamp': results['config']['timestamp'],
            'Model Type': results['config']['model_type'],
            'Used Class Weights': results['config']['use_weights'],
            'Merged KCs': results['config']['merged_codes'],
            'Embedding Info': results['config'].get('embedding_model', 'No pre-trained embeddings')
        },
        'Model Architecture': {
            'Vocab Size': results['config']['vocab_size'],
            'Embedding Dimension': results['config']['embedding_dim'],
            'Max Sequence Length': results['config']['max_length'],
            'Number of Classes': results['config']['num_classes']
        },
        'Performance Metrics': {
            'Final Validation Accuracy': f"{results['metrics']['validation_accuracy']:.4f}",
            'Final Validation Loss': f"{results['metrics']['validation_loss']:.4f}",
            'Best Validation Accuracy': f"{max(results['history']['val_accuracy']):.4f}",
            'Best Training Accuracy': f"{max(results['history']['accuracy']):.4f}",
            'Number of Epochs Trained': len(results['history']['accuracy'])
        }
    }
    
    # Create visualizations directory if it doesn't exist
    viz_dir = os.path.join(run_dir, 'visualizations/analysis')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['history']['accuracy'], label='Training')
    plt.plot(results['history']['val_accuracy'], label='Validation')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['history']['loss'], label='Training')
    plt.plot(results['history']['val_loss'], label='Validation')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'training_curves.png'))
    plt.close()
    
    # Save summary report
    summary_path = os.path.join(run_dir, 'metrics/summary', 'analysis_report.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print formatted summary
    print("\n" + "="*50)
    print(f"Analysis Report for: {os.path.basename(run_dir)}")
    print("="*50)
    
    for section, details in summary.items():
        print(f"\n{section}:")
        print("-" * len(section))
        for key, value in details.items():
            print(f"{key}: {value}")
    
    return summary

def compare_experiments(*run_dirs):
    """
    Compare multiple experiment runs.
    
    Args:
        *run_dirs: Variable number of experiment run directories
    
    Returns:
        pd.DataFrame: Comparison dataframe
    """
    comparisons = []
    
    for run_dir in run_dirs:
        # Load configuration and metrics
        with open(os.path.join(run_dir, 'models/configs', 'model_config.json'), 'r') as f:
            config = json.load(f)
        with open(os.path.join(run_dir, 'metrics/summary', 'evaluation_metrics.json'), 'r') as f:
            metrics = json.load(f)
        
        comparisons.append({
            'Run ID': os.path.basename(run_dir),
            'Model Type': config['model_type'],
            'Class Weights': 'Yes' if config['use_weights'] else 'No',
            'Merged KCs': str(config['merged_codes']),
            'Embedding': 'Yes' if 'embedding_model' in config else 'No',
            'Val Accuracy': f"{metrics['validation_accuracy']:.4f}",
            'Val Loss': f"{metrics['validation_loss']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparisons)
    
    # Save comparison to each run directory
    for run_dir in run_dirs:
        comparison_df.to_csv(os.path.join(run_dir, 'metrics/summary', 'experiment_comparison.csv'), 
                           index=False)
    
    return comparison_df

# Example usage:
if __name__ == "__main__":
    # Analyze single experiment
    run_dir = "../Results/neural_networks/lstm/weighted/run_20241210_020106"
    summary = analyze_experiment_run(run_dir)
    
    # Compare multiple experiments
    run_dir1 = "../Results/neural_networks/lstm/weighted/run_20241210_020106"
    run_dir2 = "../Results/neural_networks/cnn/unweighted/run_20241210_020205"
    comparison = compare_experiments(run_dir1, run_dir2)
    print("\nExperiment Comparison:")
    print(comparison) 