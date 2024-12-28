import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from datasets import Dataset as HFDataset
import re
import transformers

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def create_directory_structure(base_dir):
    """Create the necessary directory structure for experiment results."""
    subdirs = [
        'metrics/confusion_matrices',
        'metrics/training_history',
        'metrics/summary',
        'models/checkpoints',
        'models/configs',
        'visualizations/training_curves',
        'logs/training_logs'
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def plot_training_history(history, run_dir):
    """Plot and save training metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    train_loss = [log['loss'] for log in history if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in history if 'eval_loss' in log]
    plt.plot(train_loss, label='Training')
    plt.plot(eval_loss, label='Validation')
    plt.title('Loss Over Time')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot evaluation metrics
    plt.subplot(1, 2, 2)
    metrics = ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall']
    for metric in metrics:
        values = [log[metric] for log in history if metric in log]
        plt.plot(values, label=metric.split('_')[1].capitalize())
    plt.title('Metrics Over Time')
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    vis_dir = os.path.join(run_dir, 'visualizations', 'training_curves')
    os.makedirs(vis_dir, exist_ok=True)
    
    plt.savefig(os.path.join(vis_dir, 'training_history.png'))
    plt.close()

def run_transformer_experiment(data, merge_codes=None, model_name='roberta-base', 
                            use_weights=True, num_epochs=10):
    """
    Run transformer-based experiments with RoBERTa.
    
    Args:
        data (pd.DataFrame): Input dataframe containing 'event_result' and 'KC' columns
        merge_codes (list of lists, optional): List of KC codes to merge
        model_name (str): Name of the pre-trained model to use
        use_weights (bool): Whether to use class weights
        num_epochs (int): Number of training epochs
    
    Returns:
        str: Path to results directory
    """
    # Create a copy of the data
    df = data.copy()
    
    # Merge KCs if specified
    if merge_codes:
        for codes_to_merge in merge_codes:
            new_kc = f"KC_{'_'.join(map(str, codes_to_merge))}"
            df.loc[df['KC'].isin(codes_to_merge), 'KC'] = new_kc
    
    # Text preprocessing
    def preprocess_text(text):
        # Convert to lowercase
        text = str(text).lower()
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    # Apply preprocessing
    df['event_result'] = df['event_result'].apply(preprocess_text)
    
    # Convert KC labels to strings
    df['KC'] = df['KC'].astype(str)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['KC'])
    
    # Split the data with stratification
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']  # Ensure balanced split
    )
    
    # Create HuggingFace datasets
    train_dataset = HFDataset.from_pandas(train_df[['event_result', 'label']])
    val_dataset = HFDataset.from_pandas(val_df[['event_result', 'label']])
    
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2
    )
    
    # Tokenization function with improved settings
    def tokenize_function(examples):
        return tokenizer(
            examples['event_result'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors=None,
            add_special_tokens=True
        )
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Compute class weights if needed
    if use_weights:
        class_counts = np.bincount(train_df['label'])
        total_samples = len(train_df)
        class_weights = torch.FloatTensor(
            [total_samples / (len(class_counts) * count) for count in class_counts]
        )
    else:
        class_weights = None
    
    # Setup directory structure
    weight_str = 'weighted' if use_weights else 'unweighted'
    merge_str = '_merged' if merge_codes else ''
    results_dir = f'../Results/transformers/roberta/{weight_str}{merge_str}'
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_dir, f'run_{run_timestamp}')
    
    # Create directory structure
    create_directory_structure(run_dir)
    
    # Training arguments with improved settings
    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, 'models/checkpoints'),
        evaluation_strategy='steps',
        eval_steps=50,  # More frequent evaluation
        save_steps=50,
        learning_rate=3e-5,  # Slightly higher learning rate
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=2,
        warmup_steps=100,  # Add warmup steps
        fp16=True,  # Enable mixed precision training
        gradient_accumulation_steps=2,  # Gradient accumulation for larger effective batch size
        logging_steps=10,
        report_to=['tensorboard'],  # Enable TensorBoard logging
    )
    
    # Initialize trainer with improved settings
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.01
            ),
            # Add learning rate scheduler callback
            transformers.trainer_callback.ProgressCallback(),
        ]
    )
    
    # Train the model
    train_result = trainer.train()
    
    # Save training history with improved metrics
    history = {
        'train_loss': trainer.state.log_history,
        'eval_loss': [],
        'eval_accuracy': [],
        'eval_f1': [],
        'eval_precision': [],
        'eval_recall': []
    }
    
    for log in trainer.state.log_history:
        if 'eval_loss' in log:
            history['eval_loss'].append(log['eval_loss'])
            history['eval_accuracy'].append(log['eval_accuracy'])
            history['eval_f1'].append(log['eval_f1'])
            if 'eval_precision' in log:
                history['eval_precision'].append(log['eval_precision'])
            if 'eval_recall' in log:
                history['eval_recall'].append(log['eval_recall'])
    
    # Plot and save training curves
    plot_training_history(history, run_dir)
    
    # Save training history
    with open(os.path.join(run_dir, 'metrics/training_history', 'history.json'), 'w') as f:
        json.dump(convert_to_serializable(history), f, indent=4)
    
    # Save model configuration
    config = {
        'timestamp': run_timestamp,
        'model_name': model_name,
        'merged_codes': merge_codes if merge_codes else 'None',
        'use_weights': use_weights,
        'num_epochs': num_epochs,
        'num_classes': len(label_encoder.classes_),
        'class_mapping': dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))),
        'training_args': training_args.to_dict()
    }
    
    with open(os.path.join(run_dir, 'models/configs', 'model_config.json'), 'w') as f:
        json.dump(convert_to_serializable(config), f, indent=4)
    
    # Final evaluation
    eval_results = trainer.evaluate()
    
    # Save evaluation metrics
    with open(os.path.join(run_dir, 'metrics/summary', 'evaluation_metrics.json'), 'w') as f:
        json.dump(convert_to_serializable(eval_results), f, indent=4)
    
    return run_dir

# Example usage
if __name__ == "__main__":
    data_path = '../Data/Cleaned_Mel_CPS_19_Dataset.csv'
    data = pd.read_csv(data_path)
    
    # Run RoBERTa experiment with class weights
    run_dir = run_transformer_experiment(
        data,
        use_weights=True,
        num_epochs=10
    )
    print(f"Experiment results saved in: {run_dir}")