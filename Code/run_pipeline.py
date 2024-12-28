import pandas as pd
import os
import json
from datetime import datetime
from ml_experiments import run_ml_experiment
from dl_experiments import run_dl_experiment
from transformer_experiments import run_transformer_experiment
import openai
from pathlib import Path
import numpy as np

def load_data():
    """Load and preprocess the dataset."""
    data_path = '../Data/Cleaned_Mel_CPS_19_Dataset.csv'
    return pd.read_csv(data_path)

def run_full_pipeline(data, use_weights=True, merge_codes=None, 
                     embedding_path='/content/drive/MyDrive/crawl-300d-2M-subword.bin',
                     max_length=100):
    """Run all experiments in the pipeline."""
    results = {}
    
    # Run traditional ML experiments
    print("\nRunning Traditional ML Experiments...")
    ml_results_dir = run_ml_experiment(
        data,
        merge_codes=merge_codes,
        use_weights=use_weights
    )
    results['traditional_ml'] = ml_results_dir
    
    # Run deep learning experiments (LSTM)
    print("\nRunning Deep Learning (LSTM) Experiments...")
    lstm_results_dir = run_dl_experiment(
        data,
        model_type='lstm',
        merge_codes=merge_codes,
        use_weights=use_weights,
        embedding_path=embedding_path,
        max_length=max_length
    )
    results['lstm'] = lstm_results_dir
    
    # Run transformer experiments
    print("\nRunning Transformer (RoBERTa) Experiments...")
    transformer_results_dir = run_transformer_experiment(
        data,
        merge_codes=merge_codes,
        use_weights=use_weights
    )
    results['transformer'] = transformer_results_dir
    
    return results

def collect_metrics(results_dirs):
    """Collect metrics from all experiments."""
    metrics = {}
    
    for model_type, results_dir in results_dirs.items():
        metrics[model_type] = {}
        
        # Load model performance metrics
        if model_type == 'traditional_ml':
            performance_path = os.path.join(results_dir, 'metrics/summary/model_performance.csv')
            if os.path.exists(performance_path):
                metrics[model_type]['performance'] = pd.read_csv(performance_path).to_dict('records')
        
        # Load evaluation metrics for deep learning models
        if model_type in ['lstm', 'transformer']:
            metrics_path = os.path.join(results_dir, 'metrics/summary/evaluation_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    eval_metrics = json.load(f)
                    # Convert any nested dictionaries to strings for JSON serialization
                    metrics[model_type]['evaluation'] = convert_metrics_to_serializable(eval_metrics)
    
    return metrics

def convert_metrics_to_serializable(metrics):
    """Convert metrics to a serializable format."""
    if isinstance(metrics, dict):
        return {k: convert_metrics_to_serializable(v) for k, v in metrics.items()}
    elif isinstance(metrics, list):
        return [convert_metrics_to_serializable(item) for item in metrics]
    elif isinstance(metrics, (np.integer, np.floating)):
        return float(metrics)
    elif isinstance(metrics, np.ndarray):
        return metrics.tolist()
    else:
        return metrics

def generate_report(metrics, openai_api_key):
    """Generate a comprehensive report using GPT-4."""
    openai.api_key = openai_api_key
    
    # Convert metrics to a more readable format
    formatted_metrics = json.dumps(metrics, indent=2)
    
    # Prepare the prompt
    prompt = f"""
    Please analyze the following machine learning experiment results and write a comprehensive report.
    The experiments include traditional ML models, LSTM, and transformer-based approaches.
    
    Metrics:
    {formatted_metrics}
    
    Please structure the report with the following sections:
    1. Executive Summary
    2. Methodology Overview
    3. Results Analysis
       - Traditional ML Performance
       - Deep Learning Performance
       - Transformer Performance
    4. Model Comparison
    5. Recommendations
    
    Focus on key insights, performance comparisons, and practical recommendations.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI expert analyzing machine learning experiment results."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error generating report: {str(e)}"

def save_report(report, base_dir):
    """Save the generated report."""
    report_dir = os.path.join(base_dir, 'Reports')
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(report_dir, f'analysis_report_{timestamp}.md')
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report_path

def main():
    # Check for OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Run pipeline
    print("Running full pipeline...")
    results_dirs = run_full_pipeline(data)
    
    # Collect metrics
    print("\nCollecting metrics...")
    metrics = collect_metrics(results_dirs)
    
    # Generate report
    print("\nGenerating analysis report...")
    report = generate_report(metrics, openai_api_key)
    
    # Save report
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = save_report(report, base_dir)
    
    print(f"\nPipeline complete! Report saved to: {report_path}")

if __name__ == "__main__":
    main() 