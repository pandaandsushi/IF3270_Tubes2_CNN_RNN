from utils.layers import RNNCell
class RNN:
    def __init__(self, input_dim, hidden_dims, bidirectional=False, activation='tanh'):
        self.num_layers = len(hidden_dims)
        self.bidirectional = bidirectional  #if false then unidirectional
        self.activation = activation

        self.forward_cells = []
        self.backward_cells = []
        # for i in range(self.num_layers):
        #     in_dim = input_dim if i == 0 else hidden_dims[i - 1] * (2 if bidirectional else 1)
        #     self.forward_cells.append(RNNCell(in_dim, hidden_dims[i], activation))
        #     if bidirectional:
        #         self.backward_cells.append(RNNCell(in_dim, hidden_dims[i], activation))

    # def forward(self, x_seq, h0=None):
    #     pass

    # def backward(self, ...):
    #     pass



# rnn = RNN(input_dim=300, hidden_dims=[128, 64, 32], bidirectional=True)
# output = rnn.forward(x_seq)  # x_seq: (batch_size, seq_len, 300)
