import numpy as np

class ActivationFunction:
    
    
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True) 
    
    # Bonus: Swish, softplus, and ELU
    @staticmethod
    def swish(x):
        return x * ActivationFunction.sigmoid(x)
    
    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
class LossFunction:
    

    @staticmethod
    def mse(yi, y_hat):
        return np.mean((yi - y_hat) ** 2)
    
    @staticmethod
    def binCrossEntropy(yi, y_hat):
        return -np.mean(yi * np.log(y_hat) + (1 - yi) * np.log(1 - y_hat))
    
    @staticmethod
    def catCrossEntropy(yi, y_hat):
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
    
        n = yi.shape[0]
        
        loss = -np.sum(yi * np.log(y_hat)) / n

        return loss

class Derivative:
    

    @staticmethod
    def linear(x):
        return np.ones_like(x)
    
    @staticmethod
    def relu(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def sigmoid(x):
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def softmax(x):
        s = ActivationFunction.softmax(x)  
        batch_size, num_classes = s.shape

        # Buat matriks turunan untuk setiap sampel dalam batch
        jacobian = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            s_i = s[i].reshape(-1, 1)
            jacobian[i] = np.diagflat(s_i) - np.dot(s_i, s_i.T)

        return jacobian  
    
    # Bonus: Swish, softplus, and ELU
    @staticmethod
    def swish(x):
        s = ActivationFunction.sigmoid(x)
        return s + x * s * (1 - s)
    
    @staticmethod
    def softplus(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))

class RMSNorm:
    def __init__(self, dim):
        self.dim = dim
        self.scale = np.ones((1, dim))
    def __call__(self, x):
        rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True))
        return self.scale * (x / rms)