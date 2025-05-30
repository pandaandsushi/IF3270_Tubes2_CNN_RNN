import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from utils.data_loader import DataLoader
from utils.tokenizer import TextPreprocessor
from rnn.rnn_model import RNNModel
import random

def plot_individual_history(history, experiment_type, exp_name, save_dir):
    """Plot and save individual training history for each experiment"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{experiment_type} - {exp_name} Training History', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(history.history['loss'], label='Training Loss', marker='o', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s', color='red')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax3 = axes[1, 0]
    ax3.plot(history.history['accuracy'], label='Training Accuracy', marker='^', color='green')
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy
    ax4 = axes[1, 1]
    ax4.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='d', color='orange')
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save individual history plot
    history_plot_path = os.path.join(save_dir, f"history_{experiment_type}_{exp_name}.png")
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual training history saved to: {history_plot_path}")
    return history_plot_path


def create_experiment_plots(experiment_type, histories, results, save_dir):
    """Create comparison plots for each experiment type"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{experiment_type} - Training Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for exp_name, history in histories.items():
        ax1.plot(history['loss'], label=f'{exp_name}', marker='o', markersize=4)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for exp_name, history in histories.items():
        ax2.plot(history['val_loss'], label=f'{exp_name}', marker='s', markersize=4)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax3 = axes[1, 0]
    for exp_name, history in histories.items():
        ax3.plot(history['accuracy'], label=f'{exp_name}', marker='^', markersize=4)
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: F1 Score Comparison
    ax4 = axes[1, 1]
    exp_names = [result['experiment_name'] for result in results]
    f1_scores = [result['test_f1_score'] for result in results]
    colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
    
    bars = ax4.bar(exp_names, f1_scores, color=colors)
    ax4.set_title('Test F1 Score Comparison')
    ax4.set_ylabel('F1 Score')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'{experiment_type}_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {plot_path}")


def create_summary_analysis(all_results, save_dir):
    """Create summary analysis and conclusions"""
    
    # Group results by experiment type
    layer_results = [r for r in all_results if r['experiment_type'] == 'Layer_Effect']
    units_results = [r for r in all_results if r['experiment_type'] == 'Units_Effect']
    direction_results = [r for r in all_results if r['experiment_type'] == 'Direction_Effect']
    
    # Create summary report
    summary_path = os.path.join(save_dir, "rnn_experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("RNN HYPERPARAMETER EXPERIMENT SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Layer Effect Analysis
        f.write("1. PENGARUH JUMLAH LAYER RNN\n")
        f.write("-" * 30 + "\n")
        layer_results_sorted = sorted(layer_results, key=lambda x: x['test_f1_score'], reverse=True)
        for result in layer_results_sorted:
            f.write(f"  {result['experiment_name']}: F1={result['test_f1_score']:.4f}, "
                   f"Layers={result['layers']}, Val_Loss={result['final_val_loss']:.4f}\n")
        
        best_layer = layer_results_sorted[0]
        worst_layer = layer_results_sorted[-1]
        f.write(f"\nKESIMPULAN LAYER EFFECT:\n")
        f.write(f"- Konfigurasi terbaik: {best_layer['experiment_name']} dengan {best_layer['layers']} layer (F1: {best_layer['test_f1_score']:.4f})\n")
        f.write(f"- Konfigurasi terburuk: {worst_layer['experiment_name']} dengan {worst_layer['layers']} layer (F1: {worst_layer['test_f1_score']:.4f})\n")
        f.write(f"- Selisih performa: {best_layer['test_f1_score'] - worst_layer['test_f1_score']:.4f}\n")
        
        if best_layer['layers'] == 1:
            f.write("- Model dengan 1 layer memberikan performa terbaik, menunjukkan bahwa data tidak memerlukan representasi yang sangat kompleks\n")
        elif best_layer['layers'] == 2:
            f.write("- Model dengan 2 layer memberikan keseimbangan yang baik antara kompleksitas dan generalisasi\n")
        else:
            f.write("- Model dengan 3 layer memberikan performa terbaik, menunjukkan data memerlukan representasi hierarkis yang kompleks\n")
        f.write("\n")
        
        # Units Effect Analysis
        f.write("2. PENGARUH BANYAK CELL RNN PER LAYER\n")
        f.write("-" * 35 + "\n")
        units_results_sorted = sorted(units_results, key=lambda x: x['test_f1_score'], reverse=True)
        for result in units_results_sorted:
            f.write(f"  {result['experiment_name']}: F1={result['test_f1_score']:.4f}, "
                   f"Units={result['units']}, Val_Loss={result['final_val_loss']:.4f}\n")
        
        best_units = units_results_sorted[0]
        worst_units = units_results_sorted[-1]
        f.write(f"\nKESIMPULAN UNITS EFFECT:\n")
        f.write(f"- Konfigurasi terbaik: {best_units['experiment_name']} dengan units {best_units['units']} (F1: {best_units['test_f1_score']:.4f})\n")
        f.write(f"- Konfigurasi terburuk: {worst_units['experiment_name']} dengan units {worst_units['units']} (F1: {worst_units['test_f1_score']:.4f})\n")
        f.write(f"- Selisih performa: {best_units['test_f1_score'] - worst_units['test_f1_score']:.4f}\n")
        
        if "Small" in best_units['experiment_name']:
            f.write("- Unit yang lebih kecil memberikan performa terbaik, menunjukkan model tidak memerlukan kapasitas yang besar\n")
        elif "Medium" in best_units['experiment_name']:
            f.write("- Unit medium memberikan keseimbangan optimal antara kapasitas model dan overfitting\n")
        else:
            f.write("- Unit yang besar memberikan performa terbaik, menunjukkan data memerlukan representasi yang kaya\n")
        f.write("\n")
        
        # Direction Effect Analysis
        f.write("3. PENGARUH ARAH RNN (BIDIRECTIONAL VS UNIDIRECTIONAL)\n")
        f.write("-" * 50 + "\n")
        direction_results_sorted = sorted(direction_results, key=lambda x: x['test_f1_score'], reverse=True)
        for result in direction_results_sorted:
            direction_type = "Bidirectional" if result['bidirectional'] else "Unidirectional"
            f.write(f"  {direction_type}: F1={result['test_f1_score']:.4f}, "
                   f"Val_Loss={result['final_val_loss']:.4f}\n")
        
        best_direction = direction_results_sorted[0]
        worst_direction = direction_results_sorted[-1]
        f.write(f"\nKESIMPULAN DIRECTION EFFECT:\n")
        f.write(f"- Konfigurasi terbaik: {best_direction['experiment_name']} (F1: {best_direction['test_f1_score']:.4f})\n")
        f.write(f"- Konfigurasi terburuk: {worst_direction['experiment_name']} (F1: {worst_direction['test_f1_score']:.4f})\n")
        f.write(f"- Selisih performa: {best_direction['test_f1_score'] - worst_direction['test_f1_score']:.4f}\n")
        
        if best_direction['bidirectional']:
            f.write("- Bidirectional RNN memberikan performa lebih baik karena dapat menangkap konteks dari kedua arah\n")
            f.write("- Informasi masa depan membantu dalam memahami konteks yang lebih lengkap\n")
        else:
            f.write("- Unidirectional RNN memberikan performa lebih baik, mungkin karena data memiliki struktur sekuensial yang kuat\n")
            f.write("- Model lebih sederhana dan tidak mengalami overfitting\n")
        f.write("\n")
        
        # Overall Best Configuration
        all_results_sorted = sorted(all_results, key=lambda x: x['test_f1_score'], reverse=True)
        overall_best = all_results_sorted[0]
        f.write("4. KONFIGURASI TERBAIK SECARA KESELURUHAN\n")
        f.write("-" * 40 + "\n")
        f.write(f"Experiment Type: {overall_best['experiment_type']}\n")
        f.write(f"Configuration: {overall_best['experiment_name']}\n")
        f.write(f"Layers: {overall_best['layers']}\n")
        f.write(f"Units: {overall_best['units']}\n")
        f.write(f"Bidirectional: {overall_best['bidirectional']}\n")
        f.write(f"F1 Score: {overall_best['test_f1_score']:.4f}\n")
        f.write(f"Final Validation Loss: {overall_best['final_val_loss']:.4f}\n")
        f.write(f"Final Validation Accuracy: {overall_best['final_val_acc']:.4f}\n")
    
    print(f"Summary analysis saved to: {summary_path}")
    
    # Create overall comparison plot
    create_overall_comparison_plot(all_results, save_dir)


def create_overall_comparison_plot(all_results, save_dir):
    """Create overall comparison plot of all experiments"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RNN Hyperparameter Experiments - Overall Comparison', fontsize=16, fontweight='bold')
    
    # Group results by experiment type
    experiment_types = ['Layer_Effect', 'Units_Effect', 'Direction_Effect']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Plot 1: F1 Scores by Experiment Type
    ax1 = axes[0, 0]
    for i, exp_type in enumerate(experiment_types):
        type_results = [r for r in all_results if r['experiment_type'] == exp_type]
        exp_names = [r['experiment_name'] for r in type_results]
        f1_scores = [r['test_f1_score'] for r in type_results]
        
        x_pos = np.arange(len(exp_names)) + i * 0.25
        ax1.bar(x_pos, f1_scores, width=0.25, label=exp_type.replace('_', ' '), 
                color=colors[i], alpha=0.8)
        
        # Add value labels
        for j, (x, score) in enumerate(zip(x_pos, f1_scores)):
            ax1.text(x, score + 0.002, f'{score:.3f}', ha='center', va='bottom', 
                    fontsize=8, rotation=90)
    
    ax1.set_title('F1 Score Comparison Across All Experiments')
    ax1.set_ylabel('F1 Score')
    ax1.set_xlabel('Experiment Configurations')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Validation Loss vs F1 Score
    ax2 = axes[0, 1]
    for i, exp_type in enumerate(experiment_types):
        type_results = [r for r in all_results if r['experiment_type'] == exp_type]
        val_losses = [r['final_val_loss'] for r in type_results]
        f1_scores = [r['test_f1_score'] for r in type_results]
        
        ax2.scatter(val_losses, f1_scores, label=exp_type.replace('_', ' '), 
                   color=colors[i], s=100, alpha=0.7)
        
        # Add experiment name labels
        for result in type_results:
            ax2.annotate(result['experiment_name'], 
                        (result['final_val_loss'], result['test_f1_score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_title('Validation Loss vs F1 Score')
    ax2.set_xlabel('Final Validation Loss')
    ax2.set_ylabel('Test F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training vs Validation Loss (Final)
    ax3 = axes[1, 0]
    train_losses = [r['final_train_loss'] for r in all_results]
    val_losses = [r['final_val_loss'] for r in all_results]
    exp_names = [f"{r['experiment_type']}\n{r['experiment_name']}" for r in all_results]
    
    x_pos = np.arange(len(all_results))
    width = 0.35
    
    ax3.bar(x_pos - width/2, train_losses, width, label='Training Loss', alpha=0.8)
    ax3.bar(x_pos + width/2, val_losses, width, label='Validation Loss', alpha=0.8)
    
    ax3.set_title('Final Training vs Validation Loss')
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Experiments')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Performance Summary Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Summary table
    table_data = []
    for result in sorted(all_results, key=lambda x: x['test_f1_score'], reverse=True):
        table_data.append([
            result['experiment_name'],
            f"{result['layers']}",
            result['units'],
            "Yes" if result['bidirectional'] else "No",
            f"{result['test_f1_score']:.4f}"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Experiment', 'Layers', 'Units', 'Bidirectional', 'F1 Score'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    for i in range(len(table_data[0])):
        table[(1, i)].set_facecolor('#90EE90')  
        # Light green for best result
    
    ax4.set_title('Performance Summary (Sorted by F1 Score)', pad=20)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'rnn_overall_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Overall comparison plot saved to: {plot_path}")



# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

print("Starting RNN Hyperparameter Experiments...")

# --- Global Configuration ---
DATASET = "NusaX"
EMBEDDING_DIM = 100
MAX_LENGTH = 100
DROPOUT_RATE = 0.2
EPOCHS = 10
BATCH_SIZE = 32

# --- Experiment Configurations ---
# 1. Pengaruh jumlah layer RNN (3 variasi)
layer_experiments = [
    {"name": "Single_Layer", "layers": 1, "units": [64], "bidirectional": True},
    {"name": "Two_Layers", "layers": 2, "units": [64, 32], "bidirectional": True},
    {"name": "Three_Layers", "layers": 3, "units": [64, 32, 16], "bidirectional": True}
]

# 2. Pengaruh banyak cell RNN per layer (3 variasi)
units_experiments = [
    {"name": "Small_Units", "layers": 2, "units": [32, 16], "bidirectional": True},
    {"name": "Medium_Units", "layers": 2, "units": [64, 32], "bidirectional": True},
    {"name": "Large_Units", "layers": 2, "units": [128, 64], "bidirectional": True}
]

# 3. Pengaruh arah RNN (2 variasi)
direction_experiments = [
    {"name": "Unidirectional", "layers": 2, "units": [64, 32], "bidirectional": False},
    {"name": "Bidirectional", "layers": 2, "units": [64, 32], "bidirectional": True}
]

# Combine all experiments
all_experiments = {
    "Layer_Effect": layer_experiments,
    "Units_Effect": units_experiments,
    "Direction_Effect": direction_experiments
}

# --- Setup directories ---
save_dir = os.path.join("src", "rnn", "experiments")
os.makedirs(save_dir, exist_ok=True)

# --- Load and preprocess data ---
print("Loading NusaX dataset...")
data_loader = DataLoader(DATASET)
x_train_raw, x_val_raw, x_test_raw, y_train_raw, y_val_raw, y_test_raw = data_loader.load_data()

x_train_texts = x_train_raw[:, 1].astype(str)
x_val_texts = x_val_raw[:, 1].astype(str)
x_test_texts = x_test_raw[:, 1].astype(str)

print("Processing labels...")
label_encoder = LabelEncoder()
all_labels = np.concatenate([y_train_raw, y_val_raw, y_test_raw])
label_encoder.fit(all_labels)

y_train = label_encoder.transform(y_train_raw)
y_val = label_encoder.transform(y_val_raw)
y_test = label_encoder.transform(y_test_raw)

NUM_CLASSES = len(label_encoder.classes_)
print(f"Number of classes: {NUM_CLASSES}")

# Preprocess text
print("Preprocessing text data...")
tokenizer = TextPreprocessor(max_features=10000, max_length=MAX_LENGTH)
tokenizer.create_text_vectorization(x_train_texts)

x_train = tokenizer.preprocess_with_keras(x_train_texts)
x_val = tokenizer.preprocess_with_keras(x_val_texts)
x_test = tokenizer.preprocess_with_keras(x_test_texts)
vocab_size = tokenizer.get_vocab_size()

print(f"Vocabulary size: {vocab_size}")
print(f"Data shapes - Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

# --- Run Experiments ---
all_results = []
experiment_histories = {}

for experiment_type, experiments in all_experiments.items():
    print(f"\n{'='*60}")
    print(f"RUNNING {experiment_type.upper()} EXPERIMENTS")
    print(f"{'='*60}")
    
    experiment_results = []
    experiment_histories[experiment_type] = {}
    
    for exp_config in experiments:
        exp_name = exp_config["name"]
        layers = exp_config["layers"]
        units = exp_config["units"]
        bidirectional = exp_config["bidirectional"]
        
        print(f"\n--- Experiment: {exp_name} ---")
        print(f"Layers: {layers}, Units: {units}, Bidirectional: {bidirectional}")
        
        model = RNNModel(vocab_size, EMBEDDING_DIM, MAX_LENGTH, NUM_CLASSES)
        
        history = model.train_keras_model(
            x_train, y_train,
            x_val, y_val,
            rnn_layers=layers,
            rnn_units=units,
            bidirectional=bidirectional,
            dropout_rate=DROPOUT_RATE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        
        weights_path = os.path.join(save_dir, f"{experiment_type}_{exp_name}_weights.weights.h5")
        model.save_model_weights(weights_path)
        
        f1_keras, y_pred_keras = model.evaluate_model(x_test, y_test, model_type='keras')
        
        history_plot_path = plot_individual_history(history, experiment_type, exp_name, save_dir)
        
        # Store results
        result = {
            "experiment_type": experiment_type,
            "experiment_name": exp_name,
            "layers": layers,
            "units": str(units),
            "bidirectional": bidirectional,
            "final_train_loss": history.history['loss'][-1],
            "final_val_loss": history.history['val_loss'][-1],
            "final_train_acc": history.history['accuracy'][-1],
            "final_val_acc": history.history['val_accuracy'][-1],
            "test_f1_score": f1_keras,
            "epochs_trained": len(history.history['loss'])
        }
        
        experiment_results.append(result)
        all_results.append(result)
        experiment_histories[experiment_type][exp_name] = history.history
        
        report_path = os.path.join(save_dir, f"{experiment_type}_{exp_name}_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Experiment: {experiment_type} - {exp_name}\n")
            f.write(f"Configuration:\n")
            f.write(f"  Layers: {layers}\n")
            f.write(f"  Units: {units}\n")
            f.write(f"  Bidirectional: {bidirectional}\n")
            f.write(f"  Epochs: {EPOCHS}\n")
            f.write(f"  Batch Size: {BATCH_SIZE}\n")
            f.write(f"  Dropout Rate: {DROPOUT_RATE}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Test F1 Score: {f1_keras:.4f}\n")
            f.write(f"  Final Train Loss: {result['final_train_loss']:.4f}\n")
            f.write(f"  Final Val Loss: {result['final_val_loss']:.4f}\n")
            f.write(f"  Final Train Acc: {result['final_train_acc']:.4f}\n")
            f.write(f"  Final Val Acc: {result['final_val_acc']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred_keras))
        
        print(f"Results saved to: {report_path}")
    
    create_experiment_plots(experiment_type, experiment_histories[experiment_type], 
                          experiment_results, save_dir)

# --- Save overall results ---
results_df = pd.DataFrame(all_results)
results_csv_path = os.path.join(save_dir, "rnn_hyperparameter_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nAll results saved to: {results_csv_path}")

# --- Create summary analysis ---
create_summary_analysis(all_results, save_dir)

print("\n" + "="*60)
print("RNN HYPERPARAMETER EXPERIMENTS COMPLETED")
print("="*60)



if __name__ == "__main__":
    print("RNN Hyperparameter Experiment Script")
    print("This script will run systematic experiments on RNN hyperparameters")
    print("Results will be saved in src/rnn/experiments/")
