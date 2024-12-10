import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Input, Dropout, 
    Conv1D, MaxPooling1D, GlobalMaxPooling1D
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import KeyedVectors
import fasttext

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
        'models/weights',
        'models/configs',
        'visualizations/training_curves',
        'logs/training_logs'
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

def build_lstm_model(vocab_size, embedding_dim, max_length, num_classes):
    """Build LSTM model architecture."""
    inputs = Input(shape=(max_length,))
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = LSTM(100, return_sequences=True)(x)
    x = LSTM(50)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_cnn_model(vocab_size, embedding_dim, max_length, num_classes):
    """Build CNN model architecture."""
    inputs = Input(shape=(max_length,))
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def plot_training_history(history, run_dir, model_name):
    """Plot and save training history curves."""
    metrics = ['loss', 'accuracy']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'{model_name} - {metric.capitalize()} Over Time')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, 'visualizations/training_curves', 
                                f'{model_name.lower()}_{metric}.png'))
        plt.close()

def load_embedding_model(embedding_path, embedding_type='fasttext'):
    """
    Load pre-trained word embedding model.
    
    Args:
        embedding_path (str): Path to embedding file
        embedding_type (str): Type of embedding ('fasttext' or 'word2vec')
    
    Returns:
        Model object that supports word vector lookups
    """
    if embedding_type.lower() == 'fasttext':
        return fasttext.load_model(embedding_path)
    else:  # word2vec
        return KeyedVectors.load_word2vec_format(embedding_path, binary=True)

def create_embedding_matrix(word_index, embedding_model, embedding_type='fasttext', embedding_dim=300):
    """
    Create an embedding matrix from pre-trained embeddings.
    
    Args:
        word_index (dict): Word-to-index mapping from tokenizer
        embedding_model: Loaded embedding model
        embedding_type (str): Type of embedding ('fasttext' or 'word2vec')
        embedding_dim (int): Dimension of embeddings
    
    Returns:
        numpy array: Embedding matrix
    """
    vocab_size = len(word_index) + 1
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim))
    
    if embedding_type.lower() == 'fasttext':
        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_model.get_word_vector(word)
            except:
                continue
    else:  # word2vec
        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_model[word]
            except KeyError:
                continue
    
    return embedding_matrix

def build_lstm_model_with_embeddings(vocab_size, embedding_dim, max_length, num_classes, 
                                   embedding_matrix=None):
    """Build LSTM model with optional pre-trained embeddings."""
    inputs = Input(shape=(max_length,))
    
    if embedding_matrix is not None:
        x = Embedding(vocab_size, embedding_dim,
                     weights=[embedding_matrix],
                     trainable=False)(inputs)
    else:
        x = Embedding(vocab_size, embedding_dim)(inputs)
    
    x = LSTM(100, return_sequences=True)(x)
    x = LSTM(50)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_cnn_model_with_embeddings(vocab_size, embedding_dim, max_length, num_classes, 
                                  embedding_matrix=None):
    """Build CNN model with optional pre-trained embeddings."""
    inputs = Input(shape=(max_length,))
    
    if embedding_matrix is not None:
        x = Embedding(vocab_size, embedding_dim,
                     weights=[embedding_matrix],
                     trainable=False)(inputs)
    else:
        x = Embedding(vocab_size, embedding_dim)(inputs)
    
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def run_dl_experiment(data, model_type='lstm', merge_codes=None, use_weights=True,
                     embedding_path=None, embedding_type='fasttext'):
    """
    Run deep learning experiments with configurable KC merging and class weighting.
    
    Args:
        data (pd.DataFrame): Input dataframe containing 'event_result' and 'KC' columns
        model_type (str): Type of model to use ('lstm' or 'cnn')
        merge_codes (list of lists, optional): List of KC codes to merge
        use_weights (bool): Whether to use class weights
        embedding_path (str, optional): Path to pre-trained embeddings
        embedding_type (str): Type of embedding model ('fasttext' or 'word2vec')
    
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
    texts = df['event_result'].astype(str)
    
    # Convert KC labels to strings
    labels = df['KC'].astype(str)
    
    # Tokenize texts
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    max_length = 50  # Can be adjusted based on data
    X = pad_sequences(sequences, maxlen=max_length)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Convert to categorical
    y_cat = tf.keras.utils.to_categorical(y)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )
    
    # Compute class weights if needed
    class_weight_dict = None
    if use_weights:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
    
    # Setup directory structure
    weight_str = 'weighted' if use_weights else 'unweighted'
    merge_str = '_merged' if merge_codes else ''
    results_dir = f'../Results/neural_networks/{model_type}/{weight_str}{merge_str}'
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_dir, f'run_{run_timestamp}')
    
    # Create directory structure
    create_directory_structure(run_dir)
    
    # Model parameters
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    num_classes = y_cat.shape[1]
    
    # Create initial config dictionary
    config = {
        'timestamp': run_timestamp,
        'model_type': model_type,
        'merged_codes': merge_codes if merge_codes else 'None',
        'use_weights': use_weights,
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'max_length': max_length,
        'num_classes': num_classes,
        'class_weights': convert_to_serializable(class_weight_dict) if use_weights else 'None',
        'unique_classes': convert_to_serializable(label_encoder.classes_.tolist())
    }
    
    # Load embeddings if specified
    embedding_matrix = None
    if embedding_path:
        embedding_model = load_embedding_model(embedding_path, embedding_type)
        embedding_matrix = create_embedding_matrix(
            tokenizer.word_index, 
            embedding_model,
            embedding_type,
            embedding_dim
        )
        
        # Update config with embedding info
        config['embedding_model'] = {
            'path': embedding_path,
            'type': embedding_type,
            'trainable': False
        }
    
    # Save model configuration
    with open(os.path.join(run_dir, 'models/configs', 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Build model with embeddings
    if model_type.lower() == 'lstm':
        model = build_lstm_model_with_embeddings(
            vocab_size, embedding_dim, max_length, num_classes, embedding_matrix
        )
    elif model_type.lower() == 'cnn':
        model = build_cnn_model_with_embeddings(
            vocab_size, embedding_dim, max_length, num_classes, embedding_matrix
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Save training history
    history_dict = convert_to_serializable(history.history)
    with open(os.path.join(run_dir, 'metrics/training_history', 'history.json'), 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    # Plot training curves
    plot_training_history(history, run_dir, model_type.upper())
    
    # Save model weights
    model.save_weights(os.path.join(run_dir, 'models/weights', 'final_weights.h5'))
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    # Save evaluation metrics
    metrics = {
        'validation_loss': float(val_loss),
        'validation_accuracy': float(val_accuracy),
        'training_history': history_dict
    }
    
    with open(os.path.join(run_dir, 'metrics/summary', 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return run_dir