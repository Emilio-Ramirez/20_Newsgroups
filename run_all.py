#!/usr/bin/env python3
#Master script to run 20 newsgroups neural network classification

import os
import sys
import time
import json
import warnings
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.io as pio

# Configure plotly for static image export
pio.kaleido.scope.mathjax = None

# Supress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import all modules
print("=" * 80)
print("20 NEWSGROUPS NEURAL NETWORK CLASSIFICATION PIPELINE")
print("=" * 80)
print("Started at: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 80)

def run_phase(phase_num, phase_name, phase_function, *args, **kwargs):
    print("\n" + "="*80)
    print("PHASE " + str(phase_num) + ": " + phase_name)
    print("="*80)
    
    start_time = time.time()
    try:
        result = phase_function(*args, **kwargs)
        elapsed = time.time() - start_time
        print("Phase " + str(phase_num) + " completed in " + str(round(elapsed, 2)) + " seconds")
        return result
    except Exception as e:
        print("Error in Phase " + str(phase_num) + ": " + str(e))
        import traceback
        traceback.print_exc()
        return None

def main():
    # Create output directorys
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    pipeline_start = time.time()
    results = {}
    
    # ========== PHASE 1-2: Data Preprocessing & Exploration ==========
    print("\n" + "="*80)
    print("PHASES 1-2: DATA PREPROCESSING & EXPLORATION")
    print("="*80)
    
    import importlib.util
    
    # Import preprocessor
    spec1 = importlib.util.spec_from_file_location("preprocessing", "1_data_preprocessing.py")
    preprocessing = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(preprocessing)
    NewsGroupsPreprocessor = preprocessing.NewsGroupsPreprocessor
    
    # Import EDA
    spec2 = importlib.util.spec_from_file_location("eda", "2_exploratory_analysis.py")
    eda_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(eda_module)
    NewsGroupsEDA = eda_module.NewsGroupsEDA
    
    # Initialize preprocessor
    preprocessor = NewsGroupsPreprocessor(data_path="20_newsgroups")
    
    # Load and preprocess data
    print("Loading dataset...")
    texts, labels, categories = preprocessor.load_dataset()
    
    # Preprocess texts
    processed_texts, filtered_labels = preprocessor.preprocess_texts(texts, labels)
    
    # Create EDA analyzer
    eda = NewsGroupsEDA(processed_texts, filtered_labels, categories)
    
    # Generate and save EDA plots
    print("Generating exploratory analysis plots...")
    
    # Category distribution plot
    fig_dist = eda.plot_category_distribution()
    fig_dist.write_image("plots/category_distribution.png", width=1200, height=600)
    
    # Text length distribution
    fig_length = eda.plot_text_length_distribution()
    fig_length.write_image("plots/text_length_distribution.png", width=1200, height=800)
    
    # Word frequency plot
    fig_words = eda.plot_word_frequency()
    fig_words.write_image("plots/word_frequency.png", width=1000, height=600)
    
    # Class balance plot
    fig_balance = eda.plot_class_balance()
    fig_balance.write_image("plots/class_balance.png", width=1200, height=500)
    
    # Get dataset summary
    eda.print_dataset_summary()
    
    results['preprocessing'] = {
        'total_documents': len(processed_texts),
        'categories': len(categories),
        'category_names': categories
    }
    
    print("✅ Data preprocessing and exploration completed")
    
    # ========== PHASE 3: Neural Network Architecture ==========
    spec3 = importlib.util.spec_from_file_location("architecture", "3_model_architecture.py")
    architecture = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(architecture)
    NewsGroupsNeuralNetwork = architecture.NewsGroupsNeuralNetwork
    
    nn_builder = run_phase(
        3, 
        "NEURAL NETWORK ARCHITECTURE",
        lambda: NewsGroupsNeuralNetwork(input_dim=10000, num_classes=20)
    )
    
    if nn_builder:
        models = nn_builder.build_and_compile_models()
        comparison = nn_builder.compare_architectures()
        results['architectures'] = comparison.to_dict()
    
    # ========== PHASE 4: Training & Evaluation ==========
    print("\n" + "="*80)
    print("PHASE 4: TRAINING & EVALUATION")
    print("="*80)
    
    spec4 = importlib.util.spec_from_file_location("training", "4_training_evaluation.py")
    training = importlib.util.module_from_spec(spec4)
    spec4.loader.exec_module(training)
    NewsGroupsTrainer = training.NewsGroupsTrainer
    
    trainer = NewsGroupsTrainer(data_path="20_newsgroups")
    
    # Prepare data
    print("Preparing training data...")
    X_train, X_test, y_train_cat, y_test_cat, y_train_orig, y_test_orig = trainer.prepare_data(
        test_size=0.2, max_features=10000
    )
    
    # Train baseline model (reduce epochs for faster execution)
    print("Training baseline model...")
    trainer.train_model(
        'baseline', X_train, X_test, y_train_cat, y_test_cat, y_train_orig,
        epochs=10, batch_size=128, use_class_weights=True
    )
    
    # Evaluate model
    eval_results = trainer.evaluate_model('baseline', X_test, y_test_cat, y_test_orig)
    
    # Save training plots
    print("Saving training history plots...")
    fig_history = trainer.plot_training_history()
    if fig_history:
        fig_history.write_image("plots/training_history.png", width=1400, height=800)
    
    fig_comparison = trainer.plot_model_comparison()
    if fig_comparison:
        fig_comparison.write_image("plots/model_comparison.png", width=1200, height=500)
    
    # Save results
    trainer.save_results("results/training_results.json")
    
    results['training'] = {
        'test_accuracy': eval_results['test_accuracy'] if eval_results else 0,
        'test_loss': eval_results['test_loss'] if eval_results else 0,
        'f1_macro': eval_results['f1_macro'] if eval_results else 0
    }
    
    print("✅ Training and evaluation completed")
    
    # ========== PHASE 5: Confusion Matrix Analysis ==========
    if eval_results and 'predictions' in eval_results:
        print("\n" + "="*80)
        print("PHASE 5: CONFUSION MATRIX ANALYSIS")
        print("="*80)
        
        spec5 = importlib.util.spec_from_file_location("confusion", "5_confusion_analysis.py")
        confusion = importlib.util.module_from_spec(spec5)
        spec5.loader.exec_module(confusion)
        ConfusionMatrixAnalyzer = confusion.ConfusionMatrixAnalyzer
        
        analyzer = ConfusionMatrixAnalyzer(
            categories, 
            y_test_orig, 
            eval_results['predictions'], 
            "Baseline Model"
        )
        
        # Generate confusion matrix plots
        fig_cm_norm = analyzer.plot_confusion_matrix(normalize=True)
        fig_cm_norm.write_image("plots/confusion_matrix_normalized.png", width=800, height=800)
        
        fig_cm_raw = analyzer.plot_confusion_matrix(normalize=False)
        fig_cm_raw.write_image("plots/confusion_matrix_raw.png", width=800, height=800)
        
        # Per-class metrics plot
        fig_metrics = analyzer.plot_per_class_metrics()
        fig_metrics.write_image("plots/per_class_metrics.png", width=1200, height=800)
        
        # Get summary
        cm_summary = analyzer.get_confusion_summary()
        results['confusion_matrix'] = cm_summary
        
        print("✅ Confusion matrix analysis completed")
    
    # ========== PHASES 6-11: Comprehensive Analysis ==========
    print("\n" + "="*80)
    print("PHASES 6-11: COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    spec6 = importlib.util.spec_from_file_location("comprehensive", "6_11_comprehensive_analysis.py")
    comprehensive = importlib.util.module_from_spec(spec6)
    spec6.loader.exec_module(comprehensive)
    ComprehensiveAnalyzer = comprehensive.ComprehensiveAnalyzer
    
    analyzer = ComprehensiveAnalyzer(categories)
    
    # Run all comprehensive analyses
    # Phase 6: Architecture Experiments
    arch_results = analyzer.phase_6_architecture_experiments(X_train, y_train_orig, X_test, y_test_orig)
    
    # Phase 7: Cross Validation
    import numpy as np
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.hstack([y_train_orig, y_test_orig])
    cv_results = analyzer.phase_7_cross_validation(X_combined, y_combined)
    
    # Phase 8: ROC/AUC Analysis
    if eval_results and 'prediction_probabilities' in eval_results:
        roc_results = analyzer.phase_8_roc_auc_analysis(y_test_orig, eval_results['prediction_probabilities'])
    
    # Phase 9: Learning Curves
    learning_results = analyzer.phase_9_learning_curves()
    
    # Phase 10: Final Evaluation
    final_results = analyzer.phase_10_final_evaluation()
    
    # Phase 11: Generate Report
    report = analyzer.phase_11_generate_report()
    
    # Save comprehensive results (clean up circular references)
    # Extract only serializable data from analyzer.results
    clean_results = {}
    for key, value in analyzer.results.items():
        if isinstance(value, dict):
            # Extract only basic data, skip plotly figures
            clean_results[key] = {k: v for k, v in value.items() 
                                 if not hasattr(v, '_grid_ref')}  # Skip plotly objects
        else:
            clean_results[key] = value
    
    results['comprehensive_analysis'] = clean_results
    
    print("✅ Comprehensive analysis completed")
    
    # ========== FINAL SUMMARY ==========
    pipeline_elapsed = time.time() - pipeline_start
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"Total execution time: {pipeline_elapsed:.2f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save final results (ensure everything is JSON serializable)
    final_report = {
        'execution_date': datetime.now().isoformat(),
        'execution_time_seconds': float(pipeline_elapsed),
        'results': {
            'preprocessing': results.get('preprocessing', {}),
            'architectures': results.get('architectures', {}),
            'training': results.get('training', {}),
            'confusion_matrix': results.get('confusion_matrix', {}),
            'comprehensive_analysis': clean_results
        },
        'plots_generated': [
            'category_distribution.png',
            'text_length_distribution.png',
            'word_frequency.png',
            'class_balance.png',
            'training_history.png',
            'model_comparison.png',
            'confusion_matrix_normalized.png',
            'confusion_matrix_raw.png',
            'per_class_metrics.png'
        ]
    }
    
    # Convert numpy types to Python native types with circular reference protection
    def convert_to_serializable(obj, seen=None):
        if seen is None:
            seen = set()
        
        # Check for circular references
        obj_id = id(obj)
        if obj_id in seen:
            return str(obj)  # Return string representation for circular refs
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif hasattr(obj, '_grid_ref'):  # Skip plotly figures
            return f"Plotly Figure: {type(obj).__name__}"
        elif isinstance(obj, dict):
            seen.add(obj_id)
            try:
                result = {}
                for k, v in obj.items():
                    try:
                        result[k] = convert_to_serializable(v, seen)
                    except (RecursionError, ValueError):
                        result[k] = f"Non-serializable: {type(v).__name__}"
                return result
            finally:
                seen.discard(obj_id)
        elif isinstance(obj, list):
            seen.add(obj_id)
            try:
                return [convert_to_serializable(item, seen) for item in obj]
            finally:
                seen.discard(obj_id)
        elif hasattr(obj, '__dict__'):  # Objects with attributes
            return f"Object: {type(obj).__name__}"
        else:
            try:
                # Test if object is JSON serializable
                import json
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    final_report = convert_to_serializable(final_report)
    
    with open('results/pipeline_results.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print("\n📊 RESULTS SUMMARY:")
    print("-" * 50)
    if 'training' in results:
        print(f"Test Accuracy: {results['training']['test_accuracy']:.4f}")
        print(f"F1 Score (Macro): {results['training']['f1_macro']:.4f}")
    print(f"Total Documents: {results['preprocessing']['total_documents']}")
    print(f"Categories: {results['preprocessing']['categories']}")
    
    print("\n📁 OUTPUT FILES:")
    print("-" * 50)
    print("Plots saved in: ./plots/")
    print("Results saved in: ./results/")
    print("Main report: results/pipeline_results.json")
    
    print("\n✅ All phases completed successfully!")
    print("🎉 Project ready for deployment!")

if __name__ == "__main__":
    # Check if running in correct directory
    if not os.path.exists('20_newsgroups'):
        print("❌ Error: 20_newsgroups dataset not found in current directory!")
        print("Please ensure you're running from the exam1 directory.")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)