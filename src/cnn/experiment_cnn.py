import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from utils.data_loader import DataLoader
import os
import pickle

class CNNKeras:
    def __init__(self):
        # load cifar10 data
        self.data_loader = DataLoader('cifar10')
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.data_loader.load_data()
        
        # normalize data
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_val = self.x_val.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        
        print(f"Dataset loaded:")
        print(f"  Training: {self.x_train.shape} - {len(self.y_train)} samples")
        print(f"  Validation: {self.x_val.shape} - {len(self.y_val)} samples") 
        print(f"  Test: {self.x_test.shape} - {len(self.y_test)} samples")
        
        # # Verify the 4:1 ratio (40k:10k:10k)
        # assert len(self.y_train) == 40000, f"Expected 40k train samples, got {len(self.y_train)}"
        # assert len(self.y_val) == 10000, f"Expected 10k val samples, got {len(self.y_val)}"
        # assert len(self.y_test) == 10000, f"Expected 10k test samples, got {len(self.y_test)}"
        
    def base_model(self, num_conv_layers=2, filters=[32, 64], kernel_sizes=[3, 3], pooling_type='max'):
        model = keras.Sequential()
        
        # conv layer pertama
        model.add(layers.Conv2D(filters[0], kernel_sizes[0], activation='relu', 
                               input_shape=(32, 32, 3), padding='same', name='conv2d_1'))
        
        if pooling_type == 'max':
            model.add(layers.MaxPooling2D((2, 2), name='max_pooling2d_1'))
        else:
            model.add(layers.AveragePooling2D((2, 2), name='average_pooling2d_1'))
        
        # handle kalau ada lebih dari satu layer konvolusi
        for i in range(1, num_conv_layers):
            filter_idx = min(i, len(filters) - 1)
            kernel_idx = min(i, len(kernel_sizes) - 1)
            
            model.add(layers.Conv2D(filters[filter_idx], kernel_sizes[kernel_idx], 
                                   activation='relu', padding='same', name=f'conv2d_{i+1}'))
            
            if pooling_type == 'max':
                model.add(layers.MaxPooling2D((2, 2), name=f'max_pooling2d_{i+1}'))
            else:
                model.add(layers.AveragePooling2D((2, 2), name=f'average_pooling2d_{i+1}'))
        
        # Classifier
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(64, activation='relu', name='dense_1'))
        model.add(layers.Dense(10, activation='softmax', name='dense_2'))
        
        return model
    
    def train_and_evaluate(self, model, model_name, epochs=20):
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            self.x_train, self.y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            verbose=1
        )
        
        test_loss, test_accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        
        y_pred_proba = model.predict(self.x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        
        # simpan model beserta si bobot
        model.save(f'{model_name}.h5')
        
        print(f"Results for {model_name}:")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test F1-Score (macro): {f1_macro:.4f}")
        
        return {
            'model': model,
            'history': history,
            'test_accuracy': test_accuracy,
            'f1_score': f1_macro,
            'predictions': y_pred,
            'model_name': model_name
        }
    
    # eksperimen layer konvolusi
    def experiment_conv_layers(self, epochs=20):
        print(f"\n{'#'*80}")
        print("EXPERIMENT 1: PENGARUH JUMLAH LAYER KONVOLUSI")
        print(f"{'#'*80}")
        
        configs = [
            (1, [32], [3], "1_conv_layer"),
            (2, [32, 64], [3, 3], "2_conv_layers"), 
            (3, [32, 64, 128], [3, 3, 3], "3_conv_layers")
        ]
        
        results = {}
        for num_layers, filters, kernels, name in configs:
            model = self.create_base_model(num_layers, filters, kernels, 'max')
            results[name] = self.train_and_evaluate(model, name, epochs)
        
        self.plot_comparison(results, "Number of Convolutional Layers")
        self.analyze_conv_layers(results)
        return results
    
    def experiment_filters(self, epochs=20):
        print(f"\n{'#'*80}")
        print("EXPERIMENT 2: PENGARUH BANYAK FILTER PER LAYER")
        print(f"{'#'*80}")
        
        configs = [
            ([16, 32], "filters_16_32"),
            ([32, 64], "filters_32_64"),
            ([64, 128], "filters_64_128")
        ]
        
        results = {}
        for filters, name in configs:
            model = self.create_base_model(2, filters, [3, 3], 'max')
            results[name] = self.train_and_evaluate(model, name, epochs)
        
        self.plot_comparison(results, "Number of Filters")
        self.analyze_filters(results)
        return results
    
    def experiment_kernel_sizes(self, epochs=20):
        print(f"\n{'#'*80}")
        print("EXPERIMENT 3: PENGARUH UKURAN FILTER (KERNEL)")
        print(f"{'#'*80}")
        
        configs = [
            ([3, 3], "kernel_3x3"),
            ([5, 5], "kernel_5x5"),
            ([3, 5], "kernel_3x5_mixed")
        ]
        
        results = {}
        for kernels, name in configs:
            model = self.create_base_model(2, [32, 64], kernels, 'max')
            results[name] = self.train_and_evaluate(model, name, epochs)
        
        self.plot_comparison(results, "Kernel Sizes")
        self.analyze_kernels(results)
        return results
    
    def experiment_pooling(self, epochs=20):
        print(f"\n{'#'*80}")
        print("EXPERIMENT 4: PENGARUH JENIS POOLING LAYER")
        print(f"{'#'*80}")
        
        configs = [
            ('max', "max_pooling"),
            ('avg', "avg_pooling")
        ]
        
        results = {}
        for pooling, name in configs:
            model = self.create_base_model(2, [32, 64], [3, 3], pooling)
            results[name] = self.train_and_evaluate(model, name, epochs)
        
        self.plot_comparison(results, "Pooling Types")
        self.analyze_pooling(results)
        return results
    
    def plot_comparison(self, results, experiment_name):
        plt.figure(figsize=(15, 5))
        
        # Training & Validation Loss
        plt.subplot(1, 3, 1)
        for name, result in results.items():
            history = result['history']
            plt.plot(history.history['loss'], label=f'{name} (train)', linewidth=2)
            plt.plot(history.history['val_loss'], label=f'{name} (val)', linestyle='--', linewidth=2)
        plt.title(f'{experiment_name} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training & Validation Accuracy
        plt.subplot(1, 3, 2)
        for name, result in results.items():
            history = result['history']
            plt.plot(history.history['accuracy'], label=f'{name} (train)', linewidth=2)
            plt.plot(history.history['val_accuracy'], label=f'{name} (val)', linestyle='--', linewidth=2)
        plt.title(f'{experiment_name} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # F1-Score Comparison
        plt.subplot(1, 3, 3)
        names = list(results.keys())
        f1_scores = [results[name]['f1_score'] for name in names]
        bars = plt.bar(names, f1_scores, alpha=0.7)
        plt.title(f'{experiment_name} - F1-Score')
        plt.ylabel('F1-Score (macro)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, f1 in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{experiment_name.lower().replace(" ", "_")}_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_conv_layers(self, results):
        print(f"\n{'='*60}")
        print("ANALISIS: PENGARUH JUMLAH LAYER KONVOLUSI")
        print(f"{'='*60}")
        
        for name, result in results.items():
            layers = name.split('_')[0]
            print(f"{layers} layer(s): F1-Score = {result['f1_score']:.4f}")
        
        best = max(results.items(), key=lambda x: x[1]['f1_score'])
        worst = min(results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"\nKESIMPULAN:")
        print(f"- Performa terbaik: {best[0]} (F1-Score: {best[1]['f1_score']:.4f})")
        print(f"- Performa terburuk: {worst[0]} (F1-Score: {worst[1]['f1_score']:.4f})")
    
    def analyze_filters(self, results):
        """Analyze filters experiment"""
        print(f"\n{'='*60}")
        print("ANALISIS: PENGARUH BANYAK FILTER PER LAYER")
        print(f"{'='*60}")
        
        for name, result in results.items():
            filters = name.replace('filters_', '').replace('_', '-')
            print(f"Filter {filters}: F1-Score = {result['f1_score']:.4f}")
        
        best = max(results.items(), key=lambda x: x[1]['f1_score'])
        worst = min(results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"\nKESIMPULAN:")
        print(f"- Performa terbaik: {best[0]} (F1-Score: {best[1]['f1_score']:.4f})")
        print(f"- Performa terburuk: {worst[0]} (F1-Score: {worst[1]['f1_score']:.4f})")
    
    def analyze_kernels(self, results):
        print(f"\n{'='*60}")
        print("ANALISIS: PENGARUH UKURAN FILTER (KERNEL)")
        print(f"{'='*60}")
        
        for name, result in results.items():
            kernel = name.replace('kernel_', '').replace('_mixed', ' (mixed)')
            print(f"Kernel {kernel}: F1-Score = {result['f1_score']:.4f}")
        
        best = max(results.items(), key=lambda x: x[1]['f1_score'])
        worst = min(results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"\nKESIMPULAN:")
        print(f"- Performa terbaik: {best[0]} (F1-Score: {best[1]['f1_score']:.4f})")
        print(f"- Performa terburuk: {worst[0]} (F1-Score: {worst[1]['f1_score']:.4f})")
    
    def analyze_pooling(self, results):
        print(f"\n{'='*60}")
        print("ANALISIS: PENGARUH JENIS POOLING LAYER")
        print(f"{'='*60}")
        
        for name, result in results.items():
            pooling = name.replace('_pooling', ' pooling')
            print(f"{pooling.capitalize()}: F1-Score = {result['f1_score']:.4f}")
        
        best = max(results.items(), key=lambda x: x[1]['f1_score'])
        worst = min(results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"\nKESIMPULAN:")
        print(f"- Performa terbaik: {best[0]} (F1-Score: {best[1]['f1_score']:.4f})")
        print(f"- Performa terburuk: {worst[0]} (F1-Score: {worst[1]['f1_score']:.4f})")
    
    def run_all_experiments(self, epochs=20):        
        # Run experiments
        conv_results = self.experiment_conv_layers(epochs)
        filter_results = self.experiment_filters(epochs)
        kernel_results = self.experiment_kernel_sizes(epochs)
        pooling_results = self.experiment_pooling(epochs)
        
        # Save all results
        all_results = {
            'conv_layers': conv_results,
            'filters': filter_results,
            'kernel_sizes': kernel_results,
            'pooling': pooling_results
        }
        
        with open('experiment_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        # Overall summary
        self.print_overall_summary(all_results)
        
        return all_results
    
    def print_overall_summary(self, all_results):
        """Print overall summary of all experiments"""
        print(f"\n{'#'*80}")
        print("RINGKASAN KESELURUHAN EKSPERIMEN")
        print(f"{'#'*80}")
        
        all_models = {}
        for exp_name, exp_results in all_results.items():
            all_models.update(exp_results)
        
        # Find best overall model
        best_overall = max(all_models.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"Model terbaik secara keseluruhan: {best_overall[0]}")
        print(f"F1-Score: {best_overall[1]['f1_score']:.4f}")
        print(f"Test Accuracy: {best_overall[1]['test_accuracy']:.4f}")
        
        print(f"\nRingkasan per eksperimen:")
        for exp_name, exp_results in all_results.items():
            best_in_exp = max(exp_results.items(), key=lambda x: x[1]['f1_score'])
            print(f"- {exp_name}: {best_in_exp[0]} (F1: {best_in_exp[1]['f1_score']:.4f})")
        
        print(f"\nModel yang disimpan:")
        for model_name in all_models.keys():
            print(f"- {model_name}.h5")

def main():
    """Main function to run CNN experiments"""
    print("="*80)
    print("CNN HYPERPARAMETER EXPERIMENTS - CIFAR-10")
    print("="*80)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize experiments
    experiments = CNNKeras()
    
    # Run all experiments
    results = experiments.run_all_experiments(epochs=20)

if __name__ == "__main__":
    main()