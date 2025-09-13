#Import libraries for training neural networks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import json
from datetime import datetime
import time
import os

# Import custom modules
import importlib.util
import sys

# Import NewsGroupsPreprocessor
spec1 = importlib.util.spec_from_file_location("data_preprocessing", "1_data_preprocessing.py")
data_preprocessing = importlib.util.module_from_spec(spec1)
sys.modules["data_preprocessing"] = data_preprocessing
spec1.loader.exec_module(data_preprocessing)
NewsGroupsPreprocessor = data_preprocessing.NewsGroupsPreprocessor

# Import NewsGroupsNeuralNetwork
spec2 = importlib.util.spec_from_file_location("model_architecture", "3_model_architecture.py")
model_architecture = importlib.util.module_from_spec(spec2)
sys.modules["model_architecture"] = model_architecture
spec2.loader.exec_module(model_architecture)
NewsGroupsNeuralNetwork = model_architecture.NewsGroupsNeuralNetwork

class NewsGroupsTrainer:
    
    def __init__(self, data_path="20_newsgroups", random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.preprocessor = NewsGroupsPreprocessor(data_path)
        self.nn_builder = NewsGroupsNeuralNetwork(random_state=random_state)
        self.models = {}
        self.training_history = {}
        self.evaluation_results = {}
        self.categories = None
        self.label_encoder = LabelEncoder()
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def prepare_data(self, test_size=0.2, max_features=10000):
        """
        Prepare data for training
        
        Args:
            test_size (float): Test set proportion
            max_features (int): Maximum number of TF-IDF features
            
        Returns:
            tuple: Training and test data
        """
        print("Loading and preprocessing data...")
        
        # Load dataset
        texts, labels, self.categories = self.preprocessor.load_dataset()
        
        # Get dataset info
        info = self.preprocessor.get_data_info(texts, labels)
        print(f"Dataset: {info['total_documents']} documents, {info['categories']} categories")
        
        # Preprocess texts and filter corresponding labels
        processed_texts, filtered_labels = self.preprocessor.preprocess_texts(texts, labels)
        
        # Encode labels
        self.label_encoder.fit(filtered_labels)
        encoded_labels = self.label_encoder.transform(filtered_labels)
        
        # Vectorize texts
        X = self.preprocessor.vectorize_texts(processed_texts, max_features=max_features)
        
        # Create train/test split
        X_train, X_test, y_train, y_test = self.preprocessor.create_train_test_split(
            X, encoded_labels, test_size=test_size, random_state=self.random_state
        )
        
        # Convert labels to categorical
        y_train_categorical = to_categorical(y_train, num_classes=len(self.categories))
        y_test_categorical = to_categorical(y_test, num_classes=len(self.categories))
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train_categorical, y_test_categorical, y_train, y_test
    
    def compute_class_weights(self, y_train):
        """
        Compute class weights for handling class imbalance
        
        Args:
            y_train (np.array): Training labels (not categorical)
            
        Returns:
            dict: Class weights dictionary
        """
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        class_weight_dict = dict(enumerate(class_weights))
        return class_weight_dict
    
    def train_model(self, model_name, X_train, X_test, y_train, y_test, 
                   y_train_original, epochs=50, batch_size=128, 
                   use_class_weights=True, validation_split=0.2):
        """
        Train a specific model
        
        Args:
            model_name (str): Name of the model to train
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels (categorical)
            y_train_original: Original training labels for class weights
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            use_class_weights (bool): Whether to use class weights
            validation_split (float): Validation split ratio
            
        Returns:
            tuple: (trained_model, training_history)
        """
        print(f"\n{'='*50}")
        print(f"TRAINING {model_name.upper()} MODEL")
        print(f"{'='*50}")
        
        # Build models if not already done
        if not self.nn_builder.models:
            self.nn_builder.build_and_compile_models()
        
        # Get the model
        model = self.nn_builder.models[model_name]
        
        # Prepare class weights
        class_weights = None
        if use_class_weights:
            class_weights = self.compute_class_weights(y_train_original)
            print("Using class weights for balanced training")
        
        # Get callbacks
        callbacks = self.nn_builder.get_callbacks(model_name, patience=10)
        
        # Record training start time
        start_time = time.time()
        
        print(f"Starting training for {epochs} epochs...")
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Record training time
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Store the trained model and history
        self.models[model_name] = model
        self.training_history[model_name] = {
            'history': history.history,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss'])
        }
        
        return model, history
    
    def evaluate_model(self, model_name, X_test, y_test, y_test_original):
        """
        Evaluate a trained model
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test: Test features
            y_test: Test labels (categorical)
            y_test_original: Original test labels
            
        Returns:
            dict: Evaluation results
        """
        if model_name not in self.models:
            print(f"Model {model_name} not trained yet!")
            return None
        
        model = self.models[model_name]
        
        print(f"\nEvaluating {model_name} model...")
        
        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy, test_topk_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Detailed classification report
        class_report = classification_report(
            y_test_original, y_pred, 
            target_names=self.categories,
            output_dict=True
        )
        
        # Calculate precision, recall, f1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_original, y_pred, average=None
        )
        
        # Calculate macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test_original, y_pred, average='macro'
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test_original, y_pred, average='weighted'
        )
        
        # Store evaluation results
        evaluation_results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_topk_accuracy': test_topk_accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'classification_report': class_report,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'support': support.tolist()
            }
        }
        
        self.evaluation_results[model_name] = evaluation_results
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-3 Accuracy: {test_topk_accuracy:.4f}")
        print(f"Macro F1-Score: {f1_macro:.4f}")
        print(f"Weighted F1-Score: {f1_weighted:.4f}")
        
        return evaluation_results
    
    def plot_training_history(self, model_names=None):
        """
        Plot training history for one or more models
        
        Args:
            model_names (list): List of model names to plot, or None for all
        """
        if model_names is None:
            model_names = list(self.training_history.keys())
        elif isinstance(model_names, str):
            model_names = [model_names]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy',
                          'Learning Rate', 'Top-3 Accuracy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, model_name in enumerate(model_names):
            if model_name not in self.training_history:
                continue
                
            history = self.training_history[model_name]['history']
            color = colors[i % len(colors)]
            epochs = range(1, len(history['loss']) + 1)
            
            # Loss plot
            fig.add_trace(
                go.Scatter(x=list(epochs), y=history['loss'], 
                          name=f'{model_name} Train Loss', 
                          line=dict(color=color), legendgroup=model_name),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=history['val_loss'], 
                          name=f'{model_name} Val Loss', 
                          line=dict(color=color, dash='dash'), legendgroup=model_name),
                row=1, col=1
            )
            
            # Accuracy plot
            fig.add_trace(
                go.Scatter(x=list(epochs), y=history['accuracy'], 
                          name=f'{model_name} Train Acc', 
                          line=dict(color=color), legendgroup=model_name, showlegend=False),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=history['val_accuracy'], 
                          name=f'{model_name} Val Acc', 
                          line=dict(color=color, dash='dash'), legendgroup=model_name, showlegend=False),
                row=1, col=2
            )
            
            # Learning rate (if available)
            if 'lr' in history:
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=history['lr'], 
                              name=f'{model_name} LR', 
                              line=dict(color=color), legendgroup=model_name, showlegend=False),
                    row=2, col=1
                )
            
            # Top-3 accuracy
            fig.add_trace(
                go.Scatter(x=list(epochs), y=history['top_k_categorical_accuracy'], 
                          name=f'{model_name} Train Top-3', 
                          line=dict(color=color), legendgroup=model_name, showlegend=False),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=history['val_top_k_categorical_accuracy'], 
                          name=f'{model_name} Val Top-3', 
                          line=dict(color=color, dash='dash'), legendgroup=model_name, showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Training History Comparison",
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1)
        fig.update_yaxes(title_text="Top-3 Accuracy", row=2, col=2)
        
        fig.show()
    
    def compare_models(self):
        """
        Compare performance of all trained models
        
        Returns:
            pandas.DataFrame: Comparison table
        """
        if not self.evaluation_results:
            print("No models evaluated yet!")
            return None
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            training_time = self.training_history[model_name]['training_time']
            epochs_trained = self.training_history[model_name]['epochs_trained']
            
            comparison_data.append({
                'Model': model_name.capitalize(),
                'Test Accuracy': f"{results['test_accuracy']:.4f}",
                'Test Loss': f"{results['test_loss']:.4f}",
                'Top-3 Accuracy': f"{results['test_topk_accuracy']:.4f}",
                'Macro F1': f"{results['f1_macro']:.4f}",
                'Weighted F1': f"{results['f1_weighted']:.4f}",
                'Training Time (s)': f"{training_time:.2f}",
                'Epochs Trained': epochs_trained
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*100)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*100)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_model_comparison(self):
        """
        Create visualization comparing model performance
        """
        if not self.evaluation_results:
            print("No models to compare!")
            return
        
        # Prepare data for plotting
        model_names = list(self.evaluation_results.keys())
        accuracies = [self.evaluation_results[name]['test_accuracy'] for name in model_names]
        f1_scores = [self.evaluation_results[name]['f1_weighted'] for name in model_names]
        training_times = [self.training_history[name]['training_time'] for name in model_names]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Test Accuracy', 'Weighted F1-Score', 'Training Time'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(x=[name.capitalize() for name in model_names], y=accuracies, 
                  name='Accuracy', marker_color='lightblue'),
            row=1, col=1
        )
        
        # F1-Score comparison
        fig.add_trace(
            go.Bar(x=[name.capitalize() for name in model_names], y=f1_scores, 
                  name='F1-Score', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Training time comparison
        fig.add_trace(
            go.Bar(x=[name.capitalize() for name in model_names], y=training_times, 
                  name='Training Time', marker_color='lightcoral'),
            row=1, col=3
        )
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            height=500,
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="F1-Score", row=1, col=2)
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=3)
        
        fig.show()
    
    def save_results(self, filepath="training_results.json"):
        """
        Save training and evaluation results to file
        
        Args:
            filepath (str): Path to save results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'categories': self.categories,
            'training_history': {},
            'evaluation_results': {}
        }
        
        # Convert training history (remove non-serializable objects)
        for model_name, history_data in self.training_history.items():
            results['training_history'][model_name] = {
                'history': history_data['history'],
                'training_time': history_data['training_time'],
                'epochs_trained': history_data['epochs_trained']
            }
        
        # Convert evaluation results
        for model_name, eval_data in self.evaluation_results.items():
            results['evaluation_results'][model_name] = {
                'test_loss': float(eval_data['test_loss']),
                'test_accuracy': float(eval_data['test_accuracy']),
                'test_topk_accuracy': float(eval_data['test_topk_accuracy']),
                'precision_macro': float(eval_data['precision_macro']),
                'recall_macro': float(eval_data['recall_macro']),
                'f1_macro': float(eval_data['f1_macro']),
                'precision_weighted': float(eval_data['precision_weighted']),
                'recall_weighted': float(eval_data['recall_weighted']),
                'f1_weighted': float(eval_data['f1_weighted']),
                'per_class_metrics': eval_data['per_class_metrics']
            }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")

def main():
    """
    Main function to demonstrate training and evaluation
    """
    print("=" * 80)
    print("PHASE 4: TRAINING & EVALUATION")
    print("=" * 80)
    
    # Initialize trainer
    trainer = NewsGroupsTrainer()
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train_cat, y_test_cat, y_train_orig, y_test_orig = trainer.prepare_data(
        test_size=0.2, max_features=10000
    )
    
    # Define models to train (start with baseline for faster testing)
    models_to_train = ['baseline']  # Can add more: ['baseline', 'deep', 'wide', 'residual']
    
    print(f"\nTraining {len(models_to_train)} models...")
    
    # Train models
    for model_name in models_to_train:
        trainer.train_model(
            model_name, X_train, X_test, y_train_cat, y_test_cat, y_train_orig,
            epochs=5, batch_size=128, use_class_weights=True
        )
        
        # Evaluate model
        trainer.evaluate_model(model_name, X_test, y_test_cat, y_test_orig)
    
    # Display results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    # Compare models
    comparison = trainer.compare_models()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Plot model comparison
    trainer.plot_model_comparison()
    
    # Save results
    trainer.save_results()
    
    print("\n" + "="*50)
    print("PHASE 4 COMPLETE")
    print("="*50)
    print(f"✅ Successfully trained and evaluated {len(models_to_train)} models")
    print("✅ Training history visualized")
    print("✅ Model performance compared")
    print("✅ Results saved to training_results.json")
    print("\nReady for Phase 5: Confusion Matrix Analysis")

if __name__ == "__main__":
    main()