import numpy as np
from utils.layers import RNNCell
class RNN:
    def __init__(self, input_dim, hidden_dims, bidirectional=False, activation='tanh'):
        self.num_layers = len(hidden_dims)
        self.bidirectional = bidirectional  #if false then unidirectional
        self.activation = activation
        self.hidden_dims = hidden_dims

        self.forward_cells = []
        self.backward_cells = []

        # dimensions for each layer
        current_dim = input_dim

        for i in range(self.num_layers):
            Wx = np.random.randn(current_dim, hidden_dims[i]) * 0.01
            Wh = np.random.randn(hidden_dims[i], hidden_dims[i]) * 0.01
            b = np.zeros((1, hidden_dims[i]))
            
            self.forward_cells.append(RNNCell(Wx, Wh, b, activation))

        if self.bidirectional:
                Wx_back = np.random.randn(current_dim, hidden_dims[i]) * 0.01
                Wh_back = np.random.randn(hidden_dims[i], hidden_dims[i]) * 0.01
                b_back = np.zeros((1, hidden_dims[i]))
                
                self.backward_cells.append(RNNCell(Wx_back, Wh_back, b_back, activation))
            
        if bidirectional:
            current_dim = hidden_dims[i] * 2
        else:
            current_dim = hidden_dims[i]

    def forward(self, x_sequence):
        batch_size, seq_length, input_dim = x_sequence.shape
        layer_input = x_sequence

        for layer in range(self.num_layers):
            hidden_dim = self.hidden_dims[layer]

            forward_cell = self.forward_cells[layer]
            outputs = np.zeros((batch_size, seq_length, hidden_dim * 2 if self.bidirectional else hidden_dim))

            # forward
            h_forward = np.zeros((batch_size, hidden_dim))
            for t in range(seq_length):
                h_forward = forward_cell.forward(layer_input[:, t, :], h_forward)
                if self.bidirectional:
                    outputs[:, t, :hidden_dim] = h_forward
                else:
                    outputs[:, t, :] = h_forward

            # backward
            if self.bidirectional:
                backward_cell = self.backward_cells[layer]
                h_backward = np.zeros((batch_size, hidden_dim))
                for t in reversed(range(seq_length)):
                    h_backward = backward_cell.forward(layer_input[:, t, :], h_backward)
                    outputs[:, t, hidden_dim:] = h_backward  # second half of the output

            # output of current layer becomes input to next
            layer_input = outputs

        return outputs



    # def backward(self, ...):
    #     pass



# rnn = RNN(input_dim=300, hidden_dims=[128, 64, 32], bidirectional=True)
# output = rnn.forward(x_seq)  # x_seq: (batch_size, seq_len, 300)
