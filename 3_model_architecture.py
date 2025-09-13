"""
Neural Network Architecture module for 20 Newsgroups classification
Implements various neural network architectures for text classification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class NewsGroupsNeuralNetwork:
    """
    Neural Network Architecture class for 20 Newsgroups dataset
    """
    
    def __init__(self, input_dim=10000, num_classes=20, random_state=42):
        """
        Initialize the neural network architecture builder
        
        Args:
            input_dim (int): Input dimension (TF-IDF features)
            num_classes (int): Number of output classes
            random_state (int): Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.random_state = random_state
        self.models = {}
        self.history = {}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
    def create_baseline_model(self, dropout_rate=0.3, learning_rate=0.001):
        """
        Create a baseline neural network model
        
        Args:
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            keras.Model: Compiled neural network model
        """
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(self.input_dim,), name='dense_1'),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu', name='dense_2'),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu', name='dense_3'),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def create_deep_model(self, dropout_rate=0.4, learning_rate=0.001):
        """
        Create a deeper neural network model
        
        Args:
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            keras.Model: Compiled neural network model
        """
        model = models.Sequential([
            layers.Dense(1024, activation='relu', input_shape=(self.input_dim,), name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(512, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(256, activation='relu', name='dense_3'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(128, activation='relu', name='dense_4'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(64, activation='relu', name='dense_5'),
            layers.Dropout(dropout_rate/2),
            
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def create_wide_model(self, dropout_rate=0.3, learning_rate=0.001):
        """
        Create a wider neural network model
        
        Args:
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            keras.Model: Compiled neural network model
        """
        model = models.Sequential([
            layers.Dense(2048, activation='relu', input_shape=(self.input_dim,), name='dense_1'),
            layers.Dropout(dropout_rate),
            layers.Dense(1024, activation='relu', name='dense_2'),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu', name='dense_3'),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def create_residual_model(self, dropout_rate=0.3, learning_rate=0.001):
        """
        Create a neural network with residual connections
        
        Args:
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            keras.Model: Compiled neural network model
        """
        inputs = layers.Input(shape=(self.input_dim,))
        
        # First dense block
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.Dropout(dropout_rate)(x)
        
        # Residual block 1
        shortcut1 = layers.Dense(256, activation=None)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation=None)(x)
        x = layers.Add()([x, shortcut1])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Residual block 2
        shortcut2 = layers.Dense(128, activation=None)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(128, activation=None)(x)
        x = layers.Add()([x, shortcut2])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def get_callbacks(self, model_name, monitor='val_loss', patience=10, reduce_lr_patience=5):
        """
        Get training callbacks for the model
        
        Args:
            model_name (str): Name of the model for file saving
            monitor (str): Metric to monitor
            patience (int): Early stopping patience
            reduce_lr_patience (int): Learning rate reduction patience
            
        Returns:
            list: List of callbacks
        """
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=f'best_{model_name}_model.h5',
                monitor=monitor,
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def build_and_compile_models(self):
        """
        Build and compile all model architectures
        
        Returns:
            dict: Dictionary of compiled models
        """
        models_dict = {
            'baseline': self.create_baseline_model(),
            'deep': self.create_deep_model(),
            'wide': self.create_wide_model(),
            'residual': self.create_residual_model()
        }
        
        self.models = models_dict
        return models_dict
    
    def print_model_summary(self, model_name=None):
        """
        Print summary of one or all models
        
        Args:
            model_name (str): Name of specific model to summarize, or None for all
        """
        if model_name and model_name in self.models:
            print(f"\n=== {model_name.upper()} MODEL SUMMARY ===")
            self.models[model_name].summary()
        else:
            for name, model in self.models.items():
                print(f"\n=== {name.upper()} MODEL SUMMARY ===")
                model.summary()
                print("-" * 70)
    
    def plot_model_architecture(self, model_name, save_path=None):
        """
        Plot model architecture
        
        Args:
            model_name (str): Name of the model to plot
            save_path (str): Path to save the plot (optional)
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        try:
            keras.utils.plot_model(
                self.models[model_name],
                to_file=save_path or f'{model_name}_architecture.png',
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=False,
                dpi=96
            )
            print(f"Model architecture saved to {save_path or f'{model_name}_architecture.png'}")
        except Exception as e:
            print(f"Could not plot model architecture: {e}")
    
    def get_model_info(self):
        """
        Get information about all models
        
        Returns:
            pandas.DataFrame: DataFrame with model information
        """
        info_list = []
        
        for name, model in self.models.items():
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            info_list.append({
                'Model': name.capitalize(),
                'Layers': len(model.layers),
                'Total Parameters': f"{total_params:,}",
                'Trainable Parameters': f"{trainable_params:,}",
                'Non-trainable Parameters': f"{non_trainable_params:,}",
                'Model Size (MB)': f"{total_params * 4 / (1024**2):.2f}"  # Approximate size
            })
        
        return pd.DataFrame(info_list)
    
    def compare_architectures(self):
        """
        Compare different architecture characteristics
        
        Returns:
            pandas.DataFrame: Comparison table
        """
        if not self.models:
            print("No models built yet. Call build_and_compile_models() first.")
            return None
        
        comparison_df = self.get_model_info()
        
        print("=" * 80)
        print("NEURAL NETWORK ARCHITECTURE COMPARISON")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        
        return comparison_df

def main():
    """
    Main function to demonstrate neural network architectures
    """
    print("=" * 80)
    print("PHASE 3: NEURAL NETWORK ARCHITECTURE")
    print("=" * 80)
    
    # Initialize neural network builder
    nn_builder = NewsGroupsNeuralNetwork(input_dim=10000, num_classes=20)
    
    print("Building neural network architectures...")
    
    # Build all model architectures
    models = nn_builder.build_and_compile_models()
    
    print(f"\nBuilt {len(models)} different architectures:")
    for name in models.keys():
        print(f"  - {name.capitalize()} model")
    
    # Compare architectures
    print("\n" + "=" * 50)
    print("ARCHITECTURE COMPARISON")
    print("=" * 50)
    comparison = nn_builder.compare_architectures()
    
    # Print detailed summaries
    print("\n" + "=" * 50)
    print("DETAILED MODEL SUMMARIES")
    print("=" * 50)
    nn_builder.print_model_summary()
    
    print("\n" + "=" * 50)
    print("ARCHITECTURE PHASE COMPLETE")
    print("=" * 50)
    print("✅ Built 4 different neural network architectures")
    print("✅ Baseline model: 3 hidden layers (512-256-128)")
    print("✅ Deep model: 5 hidden layers with batch normalization")
    print("✅ Wide model: Fewer layers with more neurons (2048-1024-512)")
    print("✅ Residual model: Skip connections for better gradient flow")
    print("\nReady for Phase 4: Training & Evaluation")

if __name__ == "__main__":
    main()