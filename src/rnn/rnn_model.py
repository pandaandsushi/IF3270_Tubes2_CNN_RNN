import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, accuracy_score
import h5py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.layers import EmbeddingLayer, DenseLayer, RNNCell, DropoutLayer
from utils.other import ActivationFunction

class RNNModel:
    def __init__(self, vocab_size, embedding_dim, max_length, num_classes):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        self.keras_model = None
        self.scratch_model = None
        self.weights = {}
        
    def train_keras_model(self, x_train, y_train, x_val, y_val, 
                         rnn_layers=2, rnn_units=[64, 32], 
                         bidirectional=True, dropout_rate=0.2, 
                         epochs=5, batch_size=32):
        
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        
        model = Sequential()
        
        model.add(Embedding(input_dim=self.vocab_size, 
                           output_dim=self.embedding_dim, 
                           input_length=self.max_length))
        
        # RNN layers
        for i in range(rnn_layers):
            return_sequences = (i < rnn_layers - 1)  # Only last layer doesn't return sequences
            
            if bidirectional:
                model.add(Bidirectional(
                    SimpleRNN(rnn_units[i], 
                             return_sequences=return_sequences,
                             activation='tanh'),
                    merge_mode='concat'
                ))
            else:
                model.add(SimpleRNN(rnn_units[i], 
                                  return_sequences=return_sequences,
                                  activation='tanh'))
            
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        print("Keras Model Architecture:")
        model.summary()
        
        # Train model
        history = model.fit(x_train, y_train_cat,
                           validation_data=(x_val, y_val_cat),
                           epochs=epochs,
                           batch_size=batch_size,
                           verbose=1)
        
        self.keras_model = model
        return history
    
    def save_model_weights(self, filepath):
        if self.keras_model is not None:
            self.keras_model.save_weights(filepath)
            print(f"Model weights saved to {filepath}")
        else:
            raise ValueError("Keras model is not trained yet.")
    
    def extract_weights_from_keras(self, rnn_layers=2, rnn_units=[64, 32], bidirectional=True):
        if self.keras_model is None:
            raise ValueError("Keras model is not trained yet.")
        
        weights = {}
        layer_idx = 0
        
        # Extract embedding weights
        embedding_layer = self.keras_model.layers[layer_idx]
        weights['embedding'] = embedding_layer.get_weights()[0]
        layer_idx += 1
        
        # Extract RNN weights
        for i in range(rnn_layers):
            if bidirectional:
                # Bidirectional layer
                bi_layer = self.keras_model.layers[layer_idx]
                forward_weights = bi_layer.forward_layer.get_weights()
                backward_weights = bi_layer.backward_layer.get_weights()
                
                weights[f'rnn_{i}_forward'] = {
                    'kernel': forward_weights[0],  # Wx
                    'recurrent_kernel': forward_weights[1],  # Wh
                    'bias': forward_weights[2]  # b
                }
                weights[f'rnn_{i}_backward'] = {
                    'kernel': backward_weights[0],
                    'recurrent_kernel': backward_weights[1],
                    'bias': backward_weights[2]
                }
            else:
                # Regular RNN layer
                rnn_layer = self.keras_model.layers[layer_idx]
                rnn_weights = rnn_layer.get_weights()
                weights[f'rnn_{i}'] = {
                    'kernel': rnn_weights[0],
                    'recurrent_kernel': rnn_weights[1],
                    'bias': rnn_weights[2]
                }
            
            layer_idx += 1
            
            # Skip dropout layer if present
            if layer_idx < len(self.keras_model.layers) and 'dropout' in self.keras_model.layers[layer_idx].name.lower():
                layer_idx += 1
        
        # Extract dense layer weights
        dense_layer = self.keras_model.layers[-1]
        dense_weights = dense_layer.get_weights()
        weights['dense'] = {
            'kernel': dense_weights[0],
            'bias': dense_weights[1]
        }
        
        self.weights = weights
        return weights
    
    def rnn_scratch(self, rnn_layers=2, rnn_units=[64, 32], bidirectional=True):
        
        self.extract_weights_from_keras(rnn_layers, rnn_units, bidirectional)
        
        self.scratch_model = RNNScratchModel(
            weights=self.weights,
            rnn_layers=rnn_layers,
            rnn_units=rnn_units,
            bidirectional=bidirectional,
            max_length=self.max_length,
            num_classes=self.num_classes
        )
        
        print("RNN from scratch model created successfully!")
    
    def evaluate_model(self, x_test, y_test, model_type='keras'):
        """Evaluate model and return F1 score"""
        
        if model_type == 'keras':
            if self.keras_model is None:
                raise ValueError("Keras model is not trained yet.")
            
            # Predict with Keras model
            y_pred_proba = self.keras_model.predict(x_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
        elif model_type == 'scratch':
            if self.scratch_model is None:
                raise ValueError("Scratch model is not created yet.")
            
            # Predict with scratch model
            y_pred = self.scratch_model.predict(x_test)
            
        else:
            raise ValueError("model_type must be 'keras' or 'scratch'")
        
        # Calculate F1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{model_type.capitalize()} Model - F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        return f1, y_pred


class RNNScratchModel:
    def __init__(self, weights, rnn_layers, rnn_units, bidirectional, max_length, num_classes):
        self.weights = weights
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.bidirectional = bidirectional
        self.max_length = max_length
        self.num_classes = num_classes
        
        # Initialize layers
        self.embedding_layer = EmbeddingLayer(
            input_dim=weights['embedding'].shape[0],
            output_dim=weights['embedding'].shape[1],
            weights=weights['embedding']
        )
        
        self.rnn_cells = self._create_rnn_cells()
        
        # Initialize dropout layers
        self.dropout_layers = []
        for _ in range(rnn_layers):
            self.dropout_layers.append(DropoutLayer(rate=0.2))  # Using same dropout rate as Keras model
        
        # Initialize dense layer
        self.dense_layer = DenseLayer(
            weight=weights['dense']['kernel'],
            bias=weights['dense']['bias'],
            activation='softmax'
        )
    
    def _create_rnn_cells(self):
        """Create RNN cells from weights"""
        cells = []
        
        for i in range(self.rnn_layers):
            if self.bidirectional:
                forward_cell = RNNCell(
                    Wx=self.weights[f'rnn_{i}_forward']['kernel'],
                    Wh=self.weights[f'rnn_{i}_forward']['recurrent_kernel'],
                    b=self.weights[f'rnn_{i}_forward']['bias'],
                    activation='tanh'
                )
                backward_cell = RNNCell(
                    Wx=self.weights[f'rnn_{i}_backward']['kernel'],
                    Wh=self.weights[f'rnn_{i}_backward']['recurrent_kernel'],
                    b=self.weights[f'rnn_{i}_backward']['bias'],
                    activation='tanh'
                )
                cells.append((forward_cell, backward_cell))
            else:
                cell = RNNCell(
                    Wx=self.weights[f'rnn_{i}']['kernel'],
                    Wh=self.weights[f'rnn_{i}']['recurrent_kernel'],
                    b=self.weights[f'rnn_{i}']['bias'],
                    activation='tanh'
                )
                cells.append(cell)
        
        return cells
    
    def predict(self, x):
        """Forward pass through the scratch RNN"""
        batch_size = x.shape[0]
        
        # Set dropout layers to evaluation mode
        for dropout_layer in self.dropout_layers:
            dropout_layer.setis_training(False)
        
        # Embedding layer
        embedded = self.embedding_layer.forward(x)  # (batch_size, seq_len, embedding_dim)
        
        # RNN layers
        current_input = embedded
        
        for layer_idx in range(self.rnn_layers):
            if self.bidirectional:
                forward_cell, backward_cell = self.rnn_cells[layer_idx]
                
                # Forward pass
                forward_outputs = self._rnn_forward_pass(current_input, forward_cell)
                
                # Backward pass
                backward_outputs = self._rnn_backward_pass(current_input, backward_cell)
                
                # Concatenate forward and backward outputs
                if layer_idx == self.rnn_layers - 1:
                    # Last layer: only take final outputs
                    layer_output = np.concatenate([
                        forward_outputs[:, -1, :],
                        backward_outputs[:, 0, :]
                    ], axis=1)
                else:
                    # Intermediate layer: concatenate all timesteps
                    layer_output = np.concatenate([forward_outputs, backward_outputs], axis=2)
            else:
                cell = self.rnn_cells[layer_idx]
                outputs = self._rnn_forward_pass(current_input, cell)
                
                if layer_idx == self.rnn_layers - 1:
                    # Last layer: only take final output
                    layer_output = outputs[:, -1, :]
                else:
                    # Intermediate layer: use all outputs
                    layer_output = outputs
            
            # dropout layer
            current_input = self.dropout_layers[layer_idx].forward(layer_output)
        
        # Dense layer
        logits = self.dense_layer.forward(current_input)
        return np.argmax(logits, axis=1)
    
    def _rnn_forward_pass(self, inputs, cell):
        # Forward pass through RNN cell
        batch_size, seq_len, input_dim = inputs.shape
        hidden_dim = cell.Wh.shape[0]
        
        h = np.zeros((batch_size, hidden_dim))
        outputs = []
        
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            h = cell.forward(x_t, h)
            outputs.append(h)
        
        return np.stack(outputs, axis=1)
    
    def _rnn_backward_pass(self, inputs, cell):
        # Backward pass through RNN cell (for bidirectional)
        batch_size, seq_len, input_dim = inputs.shape
        hidden_dim = cell.Wh.shape[0]
        
        h = np.zeros((batch_size, hidden_dim))
        outputs = []
        
        # Process sequence in reverse order
        for t in range(seq_len - 1, -1, -1):
            x_t = inputs[:, t, :]
            h = cell.forward(x_t, h)
            outputs.append(h)
        
        # Reverse outputs to match forward direction
        outputs.reverse()
        return np.stack(outputs, axis=1)