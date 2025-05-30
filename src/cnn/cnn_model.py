import numpy as np
import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from utils.data_loader import DataLoader

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
    
# X = np.random.randn(2, 28, 28, 3)

# data = DataLoader('cifar10')
# x_train, x_val, x_test, y_train, y_val, y_test = data.load_data()

# conv_layer = Conv2D(filters=8, kernel_size=3, strides=1, padding='same')
# pooling_layer = Pooling(pool_type='max', pool_size=2, strides=2, padding='same')
# flatten_layer = Flatten()

# batch_size = 32
# x_batch = x_train[:batch_size]

# x = conv_layer.forward(X) 
# print(f"Output setelah Conv2D: {x.shape}")

# x = pooling_layer.forward(x)
# print(f"Output setelah Pooling: {x.shape}")

# x = flatten_layer.forward(x)
# print(f"Output setelah Flatten: {x.shape}")