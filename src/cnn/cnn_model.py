import numpy as np
from utils.layers import DenseLayer
import tensorflow as tf
from tensorflow import keras

class CNNModel:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def load_keras_model(self, keras_model):
        self.layers = []
                
        for i, keras_layer in enumerate(keras_model.layers):
            layer_name = keras_layer.name
            layer_type = type(keras_layer).__name__
            print(f"  Layer {i+1}: {layer_type} ({layer_name})")
            
            if isinstance(keras_layer, tf.keras.layers.Conv2D):
                config = keras_layer.get_config()
                conv_layer = Conv2D(
                    filters=config['filters'],
                    kernel_size=config['kernel_size'][0],
                    strides=config['strides'][0],
                    padding=config['padding'],
                    activation=config['activation']
                )
                
                weights, bias = keras_layer.get_weights()
                conv_layer.set_weights(weights, bias)
                self.add_layer(conv_layer)
                print(f"    → Conv2D: {config['filters']} filters, {config['kernel_size']}x{config['kernel_size']} kernel")
                
            elif isinstance(keras_layer, tf.keras.layers.MaxPooling2D):
                config = keras_layer.get_config()
                pool_layer = Pooling(
                    pool_type='max',
                    pool_size=config['pool_size'][0],
                    strides=config['strides'][0],
                    padding=config['padding']
                )
                self.add_layer(pool_layer)
                print(f"    → MaxPooling2D: {config['pool_size']}x{config['pool_size']} pool")
                
            elif isinstance(keras_layer, tf.keras.layers.AveragePooling2D):
                config = keras_layer.get_config()
                pool_layer = Pooling(
                    pool_type='avg',
                    pool_size=config['pool_size'][0],
                    strides=config['strides'][0],
                    padding=config['padding']
                )
                self.add_layer(pool_layer)
                print(f"    → AveragePooling2D: {config['pool_size']}x{config['pool_size']} pool")
                
            elif isinstance(keras_layer, tf.keras.layers.Flatten):
                flatten_layer = Flatten()
                self.add_layer(flatten_layer)
                print(f"    → Flatten")
                
            elif isinstance(keras_layer, tf.keras.layers.Dense):
                config = keras_layer.get_config()
                weights, bias = keras_layer.get_weights()
                
                dense_layer = DenseLayer(
                    weight=weights,
                    bias=bias,
                    activation=config['activation']
                )
                self.add_layer(dense_layer)
                print(f"    → Dense: {config['units']} units, {config['activation']} activation")
        
        print(f"✓ Model loaded with {len(self.layers)} layers")
    
    def forward(self, X):
        output = X
        for i, layer in enumerate(self.layers):
            output = layer.forward(output)
        return output
    
    def predict(self, X, batch_size=32):
        n_samples = X.shape[0]
        predictions = []
        
        print(f"Predicting {n_samples} samples in batches of {batch_size}...")
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch = X[i:batch_end]
            
            if i % (batch_size * 10) == 0:
                print(f"  Processing batch {i//batch_size + 1}/{(n_samples-1)//batch_size + 1}")
            
            batch_pred = self.forward(batch)
            predictions.append(batch_pred)
        
        return np.vstack(predictions)
    
class Conv2D:
    def __init__(self, filters, kernel_size, strides, padding, activation='relu'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel = None
        self.bias = None 

    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def _pad_input(self, X):
        if self.padding == 'same':
            pad_h = (self.kernel_size - 1) // 2
            pad_w = (self.kernel_size - 1) // 2
            return np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        return X
    
    def _activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        return x
    
    def forward(self, X):       
        X_padded = self._pad_input(X)
        batch_size, H, W, C_in = X_padded.shape
        K_h, K_w, C_in_w, C_out = self.weights.shape
        
        out_H = (H - K_h) // self.strides + 1
        out_W = (W - K_w) // self.strides + 1
        
        output = np.zeros((batch_size, out_H, out_W, C_out))
        
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.strides
                h_end = h_start + K_h
                w_start = j * self.strides
                w_end = w_start + K_w
                
                patch = X_padded[:, h_start:h_end, w_start:w_end, :]
                
                for f in range(C_out):
                    output[:, i, j, f] = np.sum(patch * self.weights[:, :, :, f], axis=(1, 2, 3)) + self.bias[f]
        
        return self._activation(output)


class Pooling:
    def __init__(self, pool_type='max', pool_size=2, strides=2, padding='valid'):
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def _pad_input(self, X):
        if self.padding == 'same':
            pad_h = (self.pool_size - 1) // 2
            pad_w = (self.pool_size - 1) // 2
            return np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        return X
    
    def forward(self, X):
        X_padded = self._pad_input(X)
        batch_size, H, W, C = X_padded.shape
        
        out_H = (H - self.pool_size) // self.strides + 1
        out_W = (W - self.pool_size) // self.strides + 1
        
        output = np.zeros((batch_size, out_H, out_W, C))
        
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.strides
                h_end = h_start + self.pool_size
                w_start = j * self.strides
                w_end = w_start + self.pool_size
                
                patch = X_padded[:, h_start:h_end, w_start:w_end, :]
                
                if self.pool_type == 'max':
                    output[:, i, j, :] = np.max(patch, axis=(1, 2))
                else:
                    output[:, i, j, :] = np.mean(patch, axis=(1, 2))
        
        return output
        
class Flatten:
    def __init__(self):
        pass

    def forward(self, X):
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)
