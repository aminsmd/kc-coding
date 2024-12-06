# kc-coding

## Project Structure

### Results Directory

The project maintains an organized structure for experimental results:

```
Results/
├── traditional_ml/
│   ├── weighted/
│   │   └── run_YYYYMMDD_HHMMSS/
│   │       ├── metrics/
│   │       │   ├── confusion_matrices/
│   │       │   ├── classification_reports/
│   │       │   └── summary/
│   │       ├── models/
│   │       │   ├── weights/
│   │       │   ├── checkpoints/
│   │       │   └── configs/
│   │       ├── visualizations/
│   │       │   ├── feature_importance/
│   │       │   └── training_curves/
│   │       └── logs/
│   │           ├── training_logs/
│   │           └── error_logs/
│   └── unweighted/
├── neural_networks/
│   ├── feedforward/
│   ├── cnn/
│   └── rnn/
└── transformers/
    ├── bert/
    ├── roberta/
    └── distilbert/
```

### Directory Structure Explanation

- **traditional_ml/**: Classical machine learning models
  - `weighted/`: Models trained with class weights
  - `unweighted/`: Models trained without class weights
  
- **neural_networks/**: Deep learning models
  - `feedforward/`: Simple neural networks
  - `cnn/`: Convolutional neural networks
  - `rnn/`: Recurrent neural networks

- **transformers/**: Transformer-based models
  - `bert/`: BERT models
  - `roberta/`: RoBERTa models
  - `distilbert/`: DistilBERT models

### Run Directory Contents

Each experimental run creates a timestamped directory containing:

- **metrics/**: Performance measurements
  - `confusion_matrices/`: Confusion matrix plots
  - `classification_reports/`: Detailed classification metrics
  - `summary/`: Overall performance metrics

- **models/**: Model artifacts
  - `weights/`: Saved model weights
  - `checkpoints/`: Training checkpoints
  - `configs/`: Model configurations

- **visualizations/**: Visual analysis
  - `feature_importance/`: Feature importance plots
  - `training_curves/`: Learning curves and training metrics

- **logs/**: Runtime information
  - `training_logs/`: Training progress logs
  - `error_logs/`: Error and warning messages

### Usage

When implementing new models, maintain this directory structure by:

1. Creating appropriate subdirectories for your model type
2. Using timestamped run directories
3. Organizing outputs into the standard categories
4. Maintaining consistent naming conventions

This structure ensures reproducibility and easy comparison across different models and experiments.