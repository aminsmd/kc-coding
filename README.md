# kc-coding

## Machine Learning Experiments

### Traditional ML Implementation

The project includes a comprehensive machine learning pipeline implemented in `ml_experiments.py` that supports:

- Multiple classifier types (Logistic Regression, Random Forest, Naive Bayes, SVM)
- Class weighting for imbalanced datasets
- KC (Knowledge Component) code merging
- Cross-validation
- Extensive metrics and visualizations

### Running Experiments

```python
from ml_experiments import run_ml_experiment
import pandas as pd

# Load data
data = pd.read_csv('path/to/data.csv')

# Run with class weights
run_dir_1 = run_ml_experiment(data, use_weights=True)

# Run without class weights
run_dir_2 = run_ml_experiment(data, use_weights=False)

# Run with merged KC codes
run_dir_3 = run_ml_experiment(data, merge_codes=[[4,5]], use_weights=True)
```

### Project Structure

The project maintains an organized structure for experimental results:

```
Results/
├── traditional_ml/                         # Currently Implemented
│   ├── weighted/
│   │   └── run_YYYYMMDD_HHMMSS/
│   │       ├── metrics/
│   │       │   ├── confusion_matrices/      # Confusion matrix plots for each model
│   │       │   ├── classification_reports/  # Detailed metrics in JSON format
│   │       │   └── summary/
│   │       │       ├── model_performance.csv    # Summary of all models
│   │       │       └── class_distribution.json  # Class distribution analysis
│   │       ├── models/
│   │       │   └── configs/
│   │       │       ├── experiment_config.json   # Experiment parameters
│   │       │       ├── class_weights.json       # Class weighting details
│   │       │       └── full_results.json        # Complete results
│   │       ├── visualizations/
│   │       │   └── feature_importance/          # Feature importance plots
│   │       └── logs/
│   │           └── training_logs/               # Training progress logs
│   ├── weighted_merged/                         # Experiments with merged KCs
│   └── unweighted/                             # Experiments without class weights
├── neural_networks/                        # Planned Implementation
│   ├── feedforward/
│   ├── cnn/
│   └── rnn/
└── transformers/                          # Planned Implementation
    ├── bert/
    ├── roberta/
    └── distilbert/
```

### Current Implementation

#### Metrics
- **Confusion Matrices**: Visual representation of model predictions
- **Classification Reports**: Precision, recall, F1-score for each class
- **Summary Statistics**: Cross-validated performance metrics

#### Visualizations
- **Feature Importance Plots**: For supported models (Logistic Regression, SVM)
- Top features contributing to each KC classification

#### Configurations
- **Experiment Parameters**: Timestamp, KC merging details, weighting configuration
- **Class Weights**: Computed weights for handling class imbalance
- **Full Results**: Comprehensive results including all metrics and analyses

### Currently Implemented Models

1. **Logistic Regression**
   - Supports class weights
   - Includes feature importance analysis

2. **Random Forest**
   - Supports class weights
   - Handles non-linear relationships

3. **Naive Bayes**
   - Multinomial implementation
   - No class weight support

4. **Support Vector Machine (SVM)**
   - Linear implementation
   - Supports class weights
   - Includes feature importance analysis

### Planned Implementations

#### Neural Networks
- Feedforward networks for basic sequence classification
- CNNs for pattern recognition in text
- RNNs for sequential data processing

#### Transformers
- BERT-based models for contextual understanding
- RoBERTa for robust performance
- DistilBERT for efficient inference

### Current Features

- **KC Merging**: Ability to combine multiple KC codes
- **Class Weighting**: Balanced class weight computation
- **Cross-validation**: 5-fold validation with multiple metrics
- **TF-IDF Vectorization**: Text feature extraction
- **Comprehensive Logging**: Detailed training and evaluation logs

### Usage Notes

1. **Data Format**: Input data should contain 'event_result' and 'KC' columns
2. **KC Merging**: Specify KC codes to merge as nested lists (e.g., `[[4,5], [1,2]]`)
3. **Class Weights**: Toggle with `use_weights` parameter
4. **Results**: Each run creates a timestamped directory with full results

This structure ensures reproducibility and easy comparison across different models and experiments, while providing a framework for future implementations.