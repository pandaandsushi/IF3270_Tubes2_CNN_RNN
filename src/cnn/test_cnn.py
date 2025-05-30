from cnn_model import Conv2D, Flatten, Pooling, CNNModel
import tensorflow as tf
from tensorflow import keras
from utils.layers import DenseLayer
from utils.data_loader import DataLoader
from sklearn.metrics import f1_score
import numpy as np

class CNNTester:
    def __init__(self):
        self.data_loader = DataLoader('cifar10')
        _, _, self.x_test, _, _, self.y_test = self.data_loader.load_data()
        self.x_test = self.x_test.astype('float32') / 255.0
        print(f"Test data loaded: {self.x_test.shape}")
    
    def validate_model(self, model_path, test_samples=500):
        """Validate from-scratch implementation against Keras model"""
        print(f"\n{'='*80}")
        print(f"VALIDATING MODEL: {model_path}")
        print(f"{'='*80}")
        
        # Load Keras model
        try:
            keras_model = keras.models.load_model(model_path)
            print(f"Keras model loaded: {model_path}")
        except Exception as e:
            print(f"Failed to load Keras model: {e}")
            return None
        
        # Print model summary
        print("\nKeras model summary:")
        keras_model.summary()
        
        # Create from-scratch model
        scratch_model = CNNModel()
        scratch_model.load_keras_model(keras_model)
        
        # Use subset for testing
        x_test_subset = self.x_test[:test_samples]
        y_test_subset = self.y_test[:test_samples]
        
        print(f"\nTesting with {test_samples} samples...")
        
        # Get Keras predictions
        print("Getting Keras predictions...")
        keras_pred = keras_model.predict(x_test_subset, batch_size=32, verbose=0)
        keras_classes = np.argmax(keras_pred, axis=1)
        keras_f1 = f1_score(y_test_subset, keras_classes, average='macro')
        
        # Get from-scratch predictions
        print("Getting from-scratch predictions...")
        scratch_pred = scratch_model.predict(x_test_subset, batch_size=16)
        scratch_classes = np.argmax(scratch_pred, axis=1)
        scratch_f1 = f1_score(y_test_subset, scratch_classes, average='macro')
        
        # Compare results
        mse_diff = np.mean((keras_pred - scratch_pred) ** 2)
        max_diff = np.max(np.abs(keras_pred - scratch_pred))
        identical_predictions = np.array_equal(keras_classes, scratch_classes)
        
        # Print detailed comparison for first few samples
        print(f"\nSample predictions comparison (first 5 samples):")
        print("Sample | Keras Pred | Scratch Pred | Keras Class | Scratch Class | True Label")
        print("-" * 75)
        for i in range(min(5, len(keras_pred))):
            k_pred = keras_pred[i]
            s_pred = scratch_pred[i]
            k_class = keras_classes[i]
            s_class = scratch_classes[i]
            true_label = y_test_subset[i]
            print(f"{i+1:6d} | {k_pred[k_class]:10.6f} | {s_pred[s_class]:11.6f} | {k_class:11d} | {s_class:12d} | {true_label:10d}")
        
        # Print results
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"MSE difference: {mse_diff:.12f}")
        print(f"Max absolute difference: {max_diff:.12f}")
        print(f"Keras F1-Score: {keras_f1:.8f}")
        print(f"From-scratch F1-Score: {scratch_f1:.8f}")
        print(f"F1-Score difference: {abs(keras_f1 - scratch_f1):.8f}")
        print(f"Identical predictions: {identical_predictions}")
        
        # Validation status
        tolerance = 1e-6
        if mse_diff < tolerance and identical_predictions:
            print("VALIDATION PASSED - Implementation is CORRECT!")
            status = "PASS"
        elif mse_diff < 1e-4:
            print("VALIDATION PASSED with small differences - Implementation is mostly correct")
            status = "PASS"
        else:
            print("VALIDATION FAILED - Implementation needs debugging!")
            status = "FAIL"
        
        return {
            'model_name': model_path,
            'mse_diff': mse_diff,
            'max_diff': max_diff,
            'keras_f1': keras_f1,
            'scratch_f1': scratch_f1,
            'identical': identical_predictions,
            'status': status
        }
    
    def validate_all_models(self, test_samples=500):
        """Validate all trained models"""
        print(f"\n{'#'*80}")
        print("VALIDATING ALL TRAINED MODELS")
        print(f"{'#'*80}")
        
        import os
        model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
        
        if not model_files:
            print("No .h5 model files found!")
            print("Please run the training script first: python complete_cnn_implementation.py")
            return {}
        
        print(f"Found {len(model_files)} model files:")
        for f in model_files:
            print(f"  - {f}")
        
        results = {}
        for model_file in model_files:
            try:
                result = self.validate_model(model_file, test_samples)
                if result:
                    results[model_file] = result
            except Exception as e:
                print(f"Error validating {model_file}: {e}")
                import traceback
                traceback.print_exc()
        
        self.print_validation_summary(results)
        return results
    
    def print_validation_summary(self, results):
        """Print summary of all validations"""
        print(f"\n{'#'*80}")
        print("VALIDATION SUMMARY")
        print(f"{'#'*80}")
        
        if not results:
            print("No successful validations.")
            return
        
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        total = len(results)
        
        print(f"OVERALL RESULTS:")
        print(f"   Total models tested: {total}")
        print(f"   Passed validation: {passed}")
        print(f"   Failed validation: {total - passed}")
        print(f"   Success rate: {passed/total*100:.1f}%")
        
        print(f"\nDETAILED RESULTS:")
        for model_name, result in results.items():
            print(f"{model_name}:")
            print(f"    MSE difference: {result['mse_diff']:.2e}")
            print(f"    Max difference: {result['max_diff']:.2e}")
            print(f"    F1 difference: {abs(result['keras_f1'] - result['scratch_f1']):.8f}")
            print(f"    Identical predictions: {result['identical']}")
        
        if passed == total:
            print(f"   All models passed validation!")
            print(f"   CNN from-scratch implementation is CORRECT!")
        else:
            print(f"\nSome models failed validation.")
            print(f"   Please check the implementation for debugging.")

def quick_test():
    print("="*60)
    print("QUICK TEST - Creating and testing a simple model")
    print("="*60)
    
    # Create simple test model
    model = keras.Sequential([
        keras.layers.Conv2D(8, 3, activation='relu', input_shape=(32, 32, 3), padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Create dummy data
    x_dummy = np.random.random((10, 32, 32, 3))
    y_dummy = np.random.randint(0, 10, 10)
    
    # Quick training
    print("Training simple model...")
    model.fit(x_dummy, y_dummy, epochs=1, verbose=0)
    
    # Test implementation
    print("Testing implementation...")
    validator = CNNTester()
    
    # Use small subset for quick test
    x_test_small = validator.x_test[:50]
    
    # Keras prediction
    keras_pred = model.predict(x_test_small, verbose=0)
    
    # From-scratch prediction
    scratch_model = CNNModel()
    scratch_model.load_keras_model(model)
    scratch_pred = scratch_model.predict(x_test_small, batch_size=10)
    
    # Compare
    mse_diff = np.mean((keras_pred - scratch_pred) ** 2)
    max_diff = np.max(np.abs(keras_pred - scratch_pred))
    
    print(f"MSE difference: {mse_diff:.10f}")
    print(f"Max difference: {max_diff:.10f}")
    
    if mse_diff < 1e-6:
        print("Quick test PASSED!")
        return True
    else:
        print("Quick test FAILED!")
        return False

def main():
    print("="*80)
    print("CNN FROM SCRATCH - FINAL VALIDATION")
    print("="*80)
    
    # Quick test first
    print("Running quick test...")
    if not quick_test():
        print("Quick test failed. Please check implementation.")
        return
    
    print("\n" + "="*80)
    print("Quick test passed! Proceeding with full validation...")
    print("="*80)
    
    # Full validation
    validator = CNNTester()
    results = validator.validate_all_models(test_samples=1000)
    
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETED!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()