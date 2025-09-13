"""
Confusion Matrix Analysis module for 20 Newsgroups classification
Provides detailed confusion matrix analysis and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import json

class ConfusionMatrixAnalyzer:
    """
    Confusion Matrix Analysis for multi-class classification
    """
    
    def __init__(self, categories, y_true, y_pred, model_name="Model"):
        """
        Initialize confusion matrix analyzer
        
        Args:
            categories (list): List of category names
            y_true (array): True labels
            y_pred (array): Predicted labels  
            model_name (str): Name of the model
        """
        self.categories = categories
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name
        self.cm = confusion_matrix(y_true, y_pred)
        self.cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
        
    def plot_confusion_matrix(self, normalize=True, title=None):
        """
        Plot confusion matrix using Plotly
        
        Args:
            normalize (bool): Whether to normalize the confusion matrix
            title (str): Custom title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Confusion matrix plot
        """
        cm_to_plot = self.cm_normalized if normalize else self.cm
        
        if title is None:
            title = f"Confusion Matrix - {self.model_name}"
            if normalize:
                title += " (Normalized)"
        
        fig = px.imshow(
            cm_to_plot,
            x=self.categories,
            y=self.categories,
            color_continuous_scale='Blues',
            aspect='auto',
            title=title,
            labels={'x': 'Predicted', 'y': 'True', 'color': 'Value'}
        )
        
        # Add text annotations
        annotations = []
        for i in range(len(self.categories)):
            for j in range(len(self.categories)):
                value = cm_to_plot[i, j]
                text_color = 'white' if value > cm_to_plot.max() * 0.5 else 'black'
                
                if normalize:
                    text = f"{value:.2f}"
                else:
                    text = f"{value:d}"
                
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=text,
                        showarrow=False,
                        font=dict(color=text_color, size=8)
                    )
                )
        
        fig.update_layout(
            annotations=annotations,
            height=800,
            width=800,
            xaxis={'tickangle': 45},
            yaxis={'tickangle': 0}
        )
        
        return fig
    
    def get_classification_report(self):
        """
        Get detailed classification report
        
        Returns:
            dict: Classification report
        """
        return classification_report(
            self.y_true, self.y_pred,
            target_names=self.categories,
            output_dict=True
        )
    
    def analyze_misclassifications(self, top_n=10):
        """
        Analyze most common misclassifications
        
        Args:
            top_n (int): Number of top misclassifications to return
            
        Returns:
            pandas.DataFrame: Top misclassifications
        """
        misclassifications = []
        
        for i in range(len(self.categories)):
            for j in range(len(self.categories)):
                if i != j and self.cm[i, j] > 0:
                    misclassifications.append({
                        'True_Category': self.categories[i],
                        'Predicted_Category': self.categories[j],
                        'Count': self.cm[i, j],
                        'Percentage': self.cm[i, j] / self.cm[i, :].sum() * 100
                    })
        
        misclass_df = pd.DataFrame(misclassifications)
        misclass_df = misclass_df.sort_values('Count', ascending=False)
        
        return misclass_df.head(top_n)
    
    def get_per_class_metrics(self):
        """
        Get per-class precision, recall, and F1-score
        
        Returns:
            pandas.DataFrame: Per-class metrics
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None
        )
        
        # Calculate accuracy per class from confusion matrix
        class_accuracy = np.diag(self.cm) / np.sum(self.cm, axis=1)
        
        metrics_df = pd.DataFrame({
            'Category': self.categories,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Accuracy': class_accuracy,
            'Support': support
        })
        
        return metrics_df.round(4)
    
    def plot_per_class_metrics(self):
        """
        Plot per-class performance metrics
        
        Returns:
            plotly.graph_objects.Figure: Metrics comparison plot
        """
        metrics_df = self.get_per_class_metrics()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Precision', 'Recall', 'F1-Score', 'Support'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Precision
        fig.add_trace(
            go.Bar(x=metrics_df['Category'], y=metrics_df['Precision'], 
                   name='Precision', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Recall
        fig.add_trace(
            go.Bar(x=metrics_df['Category'], y=metrics_df['Recall'], 
                   name='Recall', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # F1-Score
        fig.add_trace(
            go.Bar(x=metrics_df['Category'], y=metrics_df['F1_Score'], 
                   name='F1-Score', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Support
        fig.add_trace(
            go.Bar(x=metrics_df['Category'], y=metrics_df['Support'], 
                   name='Support', marker_color='lightyellow'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f"Per-Class Performance Metrics - {self.model_name}",
            height=800,
            showlegend=False
        )
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def get_confusion_summary(self):
        """
        Get summary statistics from confusion matrix
        
        Returns:
            dict: Summary statistics
        """
        total_predictions = np.sum(self.cm)
        correct_predictions = np.trace(self.cm)
        accuracy = correct_predictions / total_predictions
        
        # Per-class accuracy
        class_accuracies = np.diag(self.cm) / np.sum(self.cm, axis=1)
        
        # Most/least accurate classes
        best_class_idx = np.argmax(class_accuracies)
        worst_class_idx = np.argmin(class_accuracies)
        
        summary = {
            'overall_accuracy': accuracy,
            'total_predictions': int(total_predictions),
            'correct_predictions': int(correct_predictions),
            'best_performing_class': {
                'name': self.categories[best_class_idx],
                'accuracy': class_accuracies[best_class_idx]
            },
            'worst_performing_class': {
                'name': self.categories[worst_class_idx],
                'accuracy': class_accuracies[worst_class_idx]
            },
            'mean_class_accuracy': np.mean(class_accuracies),
            'std_class_accuracy': np.std(class_accuracies)
        }
        
        return summary

def main():
    """
    Main function to demonstrate confusion matrix analysis
    """
    print("=" * 80)
    print("PHASE 5: CONFUSION MATRIX ANALYSIS")
    print("=" * 80)
    
    # Load results from previous training (simulation with dummy data)
    categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 
                 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
    
    # Simulate some predictions for demonstration
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, len(categories), n_samples)
    
    # Create somewhat realistic predictions (with some accuracy)
    y_pred = y_true.copy()
    # Add some random errors
    error_indices = np.random.choice(n_samples, size=int(0.3 * n_samples), replace=False)
    y_pred[error_indices] = np.random.randint(0, len(categories), len(error_indices))
    
    # Initialize analyzer
    analyzer = ConfusionMatrixAnalyzer(categories, y_true, y_pred, "Baseline Model")
    
    print("Generating confusion matrix analysis...")
    
    # Plot confusion matrices
    print("✅ Normalized confusion matrix")
    fig_norm = analyzer.plot_confusion_matrix(normalize=True)
    fig_norm.show()
    
    print("✅ Raw confusion matrix") 
    fig_raw = analyzer.plot_confusion_matrix(normalize=False)
    fig_raw.show()
    
    # Get classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    class_report = analyzer.get_classification_report()
    
    # Per-class metrics
    print("\nPER-CLASS METRICS:")
    print("-" * 40)
    metrics_df = analyzer.get_per_class_metrics()
    print(metrics_df.to_string(index=False))
    
    # Plot per-class metrics
    print("✅ Per-class metrics visualization")
    metrics_fig = analyzer.plot_per_class_metrics()
    metrics_fig.show()
    
    # Misclassification analysis
    print("\nTOP MISCLASSIFICATIONS:")
    print("-" * 40)
    misclass_df = analyzer.analyze_misclassifications()
    print(misclass_df.to_string(index=False))
    
    # Summary statistics
    print("\nCONFUSION MATRIX SUMMARY:")
    print("-" * 40)
    summary = analyzer.get_confusion_summary()
    print(f"Overall Accuracy: {summary['overall_accuracy']:.4f}")
    print(f"Best Class: {summary['best_performing_class']['name']} ({summary['best_performing_class']['accuracy']:.4f})")
    print(f"Worst Class: {summary['worst_performing_class']['name']} ({summary['worst_performing_class']['accuracy']:.4f})")
    print(f"Mean Class Accuracy: {summary['mean_class_accuracy']:.4f}")
    
    print("\n" + "="*50)
    print("PHASE 5 COMPLETE")
    print("="*50)
    print("✅ Confusion matrices generated and analyzed")
    print("✅ Per-class performance metrics calculated")
    print("✅ Misclassification patterns identified")
    print("✅ Performance summary generated")
    print("\nReady for Phase 6: Architecture Experiments")

if __name__ == "__main__":
    main()