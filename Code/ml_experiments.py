import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer

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
        'metrics/classification_reports',
        'metrics/summary',
        'models/weights',
        'models/configs',
        'visualizations/feature_importance',
        'logs/training_logs'
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

def save_feature_importance(model, vectorizer, run_dir, model_name):
    """Save feature importance analysis for supported models."""
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = {}
    
    for i, class_label in enumerate(model.classes_):
        coef = model.coef_[i]
        top_indices = np.argsort(coef)[-10:][::-1]
        
        feature_importance[f'KC_{class_label}'] = {
            feature_names[idx]: float(coef[idx])
            for idx in top_indices
        }
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(feature_importance[f'KC_{class_label}'])), 
                list(feature_importance[f'KC_{class_label}'].values()))
        plt.xticks(range(len(feature_importance[f'KC_{class_label}'])), 
                   list(feature_importance[f'KC_{class_label}'].keys()), 
                   rotation=45, ha='right')
        plt.title(f'Top Features Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'visualizations/feature_importance', 
                                f'{model_name.lower()}_importance.png'))
        plt.close()
    
    return feature_importance

def run_ml_experiment(data, merge_codes=None, use_weights=True):
    """
    Run ML experiments with configurable KC merging and class weighting.
    
    Args:
        data (pd.DataFrame): Input dataframe containing 'event_result' and 'KC' columns
        merge_codes (list of lists, optional): List of KC codes to merge. 
            Example: [[4,5], [1,2]] will merge KC 4&5 and 1&2
        use_weights (bool): Whether to use class weights
    
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
    
    # Prepare the data
    X = df['event_result']
    y = df['KC'].astype(str)
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    X_tfidf = vectorizer.fit_transform(X)
    
    # Compute class weights if needed
    class_weight_dict = None
    if use_weights:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), class_weights))
    
    # Initialize models
    models = {
        'Logistic_Regression': LogisticRegression(
            max_iter=1000,
            class_weight=class_weight_dict if use_weights else None
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight_dict if use_weights else None
        ),
        'Naive_Bayes': MultinomialNB(),
        'SVM': LinearSVC(
            max_iter=2000,
            class_weight=class_weight_dict if use_weights else None
        )
    }
    
    # Setup directory structure
    weight_str = 'weighted' if use_weights else 'unweighted'
    merge_str = '_merged' if merge_codes else ''
    results_dir = f'../Results/traditional_ml/{weight_str}{merge_str}'
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_dir, f'run_{run_timestamp}')
    
    # Create directory structure
    create_directory_structure(run_dir)
    
    # Save experiment configuration
    config = {
        'timestamp': run_timestamp,
        'merged_codes': merge_codes if merge_codes else 'None',
        'use_weights': use_weights,
        'class_weights': convert_to_serializable(class_weight_dict) if use_weights else 'None',
        'unique_classes': convert_to_serializable(np.unique(y).tolist())
    }
    with open(os.path.join(run_dir, 'models/configs', 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Initialize results storage
    all_results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # Perform cross-validation
        cv_results = cross_validate(
            model, X_tfidf, y,
            cv=5,
            scoring={
                'accuracy': 'accuracy',
                'f1_macro': 'f1_macro',
                'f1_weighted': 'f1_weighted'
            },
            return_train_score=True
        )
        
        # Train final model on full dataset
        model.fit(X_tfidf, y)
        y_pred = model.predict(X_tfidf)
        
        # Create confusion matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(10, 8))
        
        # Get unique labels for axis labels
        unique_labels = sorted(np.unique(y))
        
        # Create heatmap with proper labels
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=unique_labels,
            yticklabels=unique_labels
        )
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Create directories if they don't exist
        cm_dir = os.path.join(run_dir, 'metrics', 'confusion_matrices')
        os.makedirs(cm_dir, exist_ok=True)
        
        # Save with tight layout to prevent label cutoff
        plt.tight_layout()
        plt.savefig(os.path.join(cm_dir, f'{name.lower()}_cm.png'))
        plt.close()
        
        # Store results
        model_results = {
            'cross_validation': {
                'accuracy': {
                    'mean': float(cv_results['test_accuracy'].mean()),
                    'std': float(cv_results['test_accuracy'].std())
                },
                'f1_macro': {
                    'mean': float(cv_results['test_f1_macro'].mean()),
                    'std': float(cv_results['test_f1_macro'].std())
                },
                'f1_weighted': {
                    'mean': float(cv_results['test_f1_weighted'].mean()),
                    'std': float(cv_results['test_f1_weighted'].std())
                }
            },
            'confusion_matrix': convert_to_serializable(cm),
            'classification_report': classification_report(y, y_pred, 
                                                        output_dict=True)
        }
        
        # Add feature importance for supported models
        if name in ['Logistic_Regression', 'SVM']:
            model_results['feature_importance'] = save_feature_importance(
                model, vectorizer, run_dir, name
            )
        
        all_results[name] = model_results
        
        # Save individual classification report
        with open(os.path.join(run_dir, 'metrics/classification_reports', 
                              f'{name.lower()}_report.json'), 'w') as f:
            json.dump(model_results['classification_report'], f, indent=4)
    
    # Save full results
    with open(os.path.join(run_dir, 'models/configs', 'full_results.json'), 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=4)
    
    # Create and save summary metrics
    summary_data = []
    for model_name, results in all_results.items():
        summary_data.append({
            'Model': model_name,
            'Class_Weights': 'Yes' if use_weights and model_name != 'Naive_Bayes' else 'No',
            'Accuracy': f"{results['cross_validation']['accuracy']['mean']:.3f} ± "
                       f"{results['cross_validation']['accuracy']['std']*2:.3f}",
            'F1_Macro': f"{results['cross_validation']['f1_macro']['mean']:.3f} ± "
                       f"{results['cross_validation']['f1_macro']['std']*2:.3f}",
            'F1_Weighted': f"{results['cross_validation']['f1_weighted']['mean']:.3f} ± "
                          f"{results['cross_validation']['f1_weighted']['std']*2:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(run_dir, 'metrics/summary', 
                                  'model_performance.csv'), index=False)
    
    return run_dir