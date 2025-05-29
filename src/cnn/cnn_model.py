import numpy as np
import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from utils.data_loader import DataLoader

class Conv2D:
    def __init__(self, filters, kernel_size, strides, padding, activation='relu', kernel_init='he', bias_init='zeros'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.kernel = None
        self.bias = None 

    def init_kernel(self):
        if self.kernel_init == 'he':
            return np.random.randn(self.kernel_size, self.kernel_size, 3, self.filters) * np.sqrt(2. / (self.kernel_size * self.kernel_size * 3))
    
    def init_bias(self):
        if self.bias_init == 'zeros':
            return np.zeros((self.filters,))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def pad_input(self, X):
        if self.padding == 'same':
            pad_h = (self.kernel_size - 1) // 2
            pad_w = (self.kernel_size - 1) // 2
            return np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
        return X
    
    def forward(self, X):
        # inisialisasi bobot kernel dan bias
        if self.kernel is None:
            input_channels = X.shape[-1]
            self.kernel = self.init_kernel(input_channels)
            self.bias = self.init_bias()

        X = self.pad_input(X)
        batch_size, H, W, C = X.shape
        F, _, _, K = self.kernel.shape
        FM_height = (H - F)//self.strides + 1
        FM_width = (W - F)//self.strides + 1
        out = np.zeros((batch_size, FM_height, FM_width, K))

        for i in range(batch_size):
            for j in range(FM_height):
                for k in range(FM_width):
                    for l in range(K):
                        out[i, j, k, l] = np.sum(X[i, j*self.strides:j*self.strides+F, k*self.strides:k*self.strides+F, :] * self.kernel[:, :, :, l]) + self.bias[l]
        
        if self.activation == 'relu':
            out = self.relu(out)
        
        return out


class Pooling:
    def __init__(self, pool_type, pool_size, strides, padding):
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def pad_input(self, X):
        if self.padding == 'same':
            pad_h = (self.pool_size - 1) // 2
            pad_w = (self.pool_size - 1) // 2
            return np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
        return X
    
    def avg_pool(self, X):
        batch_size, H, W, C = X.shape
        pool_height, pool_width = self.pool_size, self.pool_size
        pool_out_height = (H - pool_height) // self.strides + 1
        pool_out_width = (W - pool_width) // self.strides + 1
        out = np.zeros((batch_size, pool_out_height, pool_out_width, C))

        for i in range(batch_size):
            for j in range(pool_out_height):
                for k in range(pool_out_width):
                    out[i, j, k, :] = np.mean(X[i, j*self.strides:j*self.strides+pool_height, k*self.strides:k*self.strides+pool_width, :], axis=(0, 1, 2))
        
        return out

    def max_pool(self, X):
        batch_size, H, W, C = X.shape
        pool_height, pool_width = self.pool_size, self.pool_size
        pool_out_height = (H - pool_height) // self.strides + 1
        pool_out_width = (W - pool_width) // self.strides + 1
        out = np.zeros((batch_size, pool_out_height, pool_out_width, C))

        for i in range(batch_size):
            for j in range(pool_out_height):
                for k in range(pool_out_width):
                    out[i, j, k, :] = np.max(X[i, j*self.strides:j*self.strides+pool_height, k*self.strides:k*self.strides+pool_width, :], axis=(0, 1, 2))
        return out

    def forward(self, X):
        X = self.pad_input(X)
        if self.pool_type == 'max':
            return self.max_pool(X)
        elif self.pool_type == 'avg':
            return self.avg_pool(X)
        
class Flatten:
    def __init__(self):
        pass

    def forward(self, X):
        batch_size, _, _, _ = X.shape
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