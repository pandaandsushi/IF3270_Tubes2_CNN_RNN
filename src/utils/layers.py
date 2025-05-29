from utils.other import ActivationFunction
import numpy as np

class EmbeddingLayer:
    def __init__(self, input_dim, output_dim, weights=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if weights is not None:
            self.embedding_matrix = weights
        else:
            # just use random if cannot read
            self.embedding_matrix = np.random.randn(input_dim, output_dim) * 0.01
    
    def forward(self, x):
        # x = (batch_size, sequence_length)
        return np.take(self.embedding_matrix, x, axis=0)

class DenseLayer:
    def __init__(self, weight, bias, activation='linear'):
        self.weight = weight
        self.bias = bias
        self.activation = activation

    def forward(self, input):
        sigma = np.dot(input, self.weight) + self.bias
        return self._activate(sigma)

    def _activate(self, sigma):
        if self.activation == 'linear':
            return ActivationFunction.linear(sigma)
        elif self.activation == 'relu':
            return ActivationFunction.relu(sigma)
        elif self.activation == 'sigmoid':
            return ActivationFunction.sigmoid(sigma)
        elif self.activation == 'tanh':
            return ActivationFunction.tanh(sigma)
        elif self.activation == 'softmax':
            return ActivationFunction.softmax(sigma)
        elif self.activation == 'swish':
            return ActivationFunction.swish(sigma)
        elif self.activation == 'softplus':
            return ActivationFunction.softplus(sigma)
        elif self.activation == 'elu':
            return ActivationFunction.elu(sigma)
        else:
            raise ValueError("Unknown activation function")
    
class RNNCell:
    def __init__(self, Wx, Wh, b, activation='tanh'):
        self.Wx = Wx
        self.Wh = Wh
        self.b = b
        self.activation = activation
        
    def _activate(self, sigma):
        if self.activation == 'linear':
            return ActivationFunction.linear(sigma)
        elif self.activation == 'relu':
            return ActivationFunction.relu(sigma)
        elif self.activation == 'sigmoid':
            return ActivationFunction.sigmoid(sigma)
        elif self.activation == 'tanh':
            return ActivationFunction.tanh(sigma)
        elif self.activation == 'softmax':
            return ActivationFunction.softmax(sigma)
        elif self.activation == 'swish':
            return ActivationFunction.swish(sigma)
        elif self.activation == 'softplus':
            return ActivationFunction.softplus(sigma)
        elif self.activation == 'elu':
            return ActivationFunction.elu(sigma)
        else:
            raise ValueError("Unknown activation function")
        
    def forward(self, x_t, h_prev):
        return self._activate(np.dot(x_t, self.Wx) + np.dot(h_prev, self.Wh) + self.b)

class LSTMCell:
    """
    Represents a single LSTM cell. 
    Takes input x_t, h_prev, and c_prev, 
    and outputs h_t and c_t.
    """
    def __init__(self, Wi, Wf, Wo, Wc,
                       Ui, Uf, Uo, Uc,
                       bi, bf, bo, bc):
        # init weights and biases
        self.Wi, self.Wf, self.Wo, self.Wc = Wi, Wf, Wo, Wc
        self.Ui, self.Uf, self.Uo, self.Uc = Ui, Uf, Uo, Uc
        self.bi, self.bf, self.bo, self.bc = bi, bf, bo, bc

    def forward(self, x_t, h_prev, c_prev):
        # input gate
        i_t = ActivationFunction.sigmoid(
            np.dot(x_t, self.Wi) + np.dot(h_prev, self.Ui) + self.bi
        )
        
        # forget gate
        f_t = ActivationFunction.sigmoid(
            np.dot(x_t, self.Wf) + np.dot(h_prev, self.Uf) + self.bf
        )
        
        # output gate
        o_t = ActivationFunction.sigmoid(
            np.dot(x_t, self.Wo) + np.dot(h_prev, self.Uo) + self.bo
        )
        
        # candidate cell state
        c_hat_t = ActivationFunction.tanh(
            np.dot(x_t, self.Wc) + np.dot(h_prev, self.Uc) + self.bc
        )

        # new cell state
        c_t = f_t * c_prev + i_t * c_hat_t
        
        # new hidden state
        h_t = o_t * ActivationFunction.tanh(c_t)

        return h_t, c_t

class DropoutLayer:
    def __init__(self, rate=0):
        self.rate = rate
        self.mask = None
        self.is_training = True

    def setrate(self,rate):
        self.rate = rate

    def setis_training(self,is_training):
        self.is_training = is_training

    def forward(self, x):
            if self.is_training and self.rate > 0:
                self.mask = (np.random.rand(*x.shape) > self.rate).astype(np.float32)
                return (x * self.mask) / (1.0 - self.rate)
            return x

    