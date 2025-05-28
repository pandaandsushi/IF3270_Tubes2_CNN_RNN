import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
from utils.layers import EmbeddingLayer, RNNCell, DenseLayer, DropoutLayer
from rnn.rnn_layer import RNNLayer

class RNNModel:
    def __init__(self, vocab_size, embedding_dim, max_length, num_classes):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        
        # storing trained models
        self.keras_model = None
        self.model_weights = {}
        
        self.embedding = None
        self.rnn = None
        self.dropout = None
        self.dense = None

    def create_keras_model(self, rnn_layers, rnn_units, bidirectional=False, dropout_rate=0.2):
        model = keras.Sequential()
        
        # Embedding layer
        model.add(layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length))
        
        # RNN layers
        for i in range(rnn_layers):
            # Return sequences for all but last layer
            return_sequences = (i < rnn_layers - 1)  
            
            if bidirectional:
                model.add(layers.Bidirectional(
                    layers.SimpleRNN(rnn_units[i], return_sequences=return_sequences, activation='tanh')
                ))
            else:
                model.add(layers.SimpleRNN(rnn_units[i], return_sequences=return_sequences, activation='tanh'))
        
        # Dropout layer
        model.add(layers.Dropout(dropout_rate))
        
        # Dense output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict_keras_model(self, x):
        if self.keras_model is None:
            raise ValueError("No trained Keras model available")
        
        return self.keras_model.predict(x)
    
    def train_keras_model(self, x_train, y_train, x_val, y_val, rnn_layers, rnn_units, 
        bidirectional=False, dropout_rate=0.2, epochs=10, batch_size=32):
        self.keras_model = self.create_keras_model(rnn_layers, rnn_units, bidirectional, dropout_rate)
        
        print(f"Training model with {rnn_layers} RNN layers, units: {rnn_units}, bidirectional: {bidirectional}")
        print(self.keras_model.summary())
        
        # Train the model
        history = self.keras_model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def save_model_weights(self, filepath):
        weights = {}
        layer_idx = 0
        if self.keras_model is None:
            raise ValueError("No trained model to save")
        
        self.keras_model.save_weights(filepath)  
        
        if self.keras_model is None:
            raise ValueError("No trained model to extract weights from")
        
        for layer in self.keras_model.layers:
            if isinstance(layer, layers.Embedding):
                weights['embedding'] = layer.get_weights()[0]
            elif isinstance(layer, (layers.SimpleRNN, layers.Bidirectional)):
                if isinstance(layer, layers.Bidirectional):
                    # bidirectional RNN
                    forward_weights = layer.forward_layer.get_weights()
                    backward_weights = layer.backward_layer.get_weights()
                    weights[f'rnn_{layer_idx}_forward'] = {
                        'Wx': forward_weights[0],
                        'Wh': forward_weights[1],
                        'b': forward_weights[2]
                    }
                    weights[f'rnn_{layer_idx}_backward'] = {
                        'Wx': backward_weights[0],
                        'Wh': backward_weights[1],
                        'b': backward_weights[2]
                    }
                else:
                    # unidirectional RNN
                    layer_weights = layer.get_weights()
                    weights[f'rnn_{layer_idx}'] = {
                        'Wx': layer_weights[0],
                        'Wh': layer_weights[1],
                        'b': layer_weights[2]
                    }
                layer_idx += 1
            elif isinstance(layer, layers.Dense):
                layer_weights = layer.get_weights()
                weights['dense'] = {
                    'W': layer_weights[0],
                    'b': layer_weights[1]
                }
        
        self.model_weights = weights

    def rnn_scratch(self, rnn_layers, rnn_units, bidirectional=False):
        if not self.model_weights:
            raise ValueError("No weights available. Train a model first.")
        
        self.embedding = EmbeddingLayer(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            weights=self.model_weights['embedding']
        )
        
        self.rnn = RNNLayer(
            input_dim=self.embedding_dim,
            hidden_dims=rnn_units,
            bidirectional=bidirectional
        )
        
        # Set RNN weights
        for i in range(rnn_layers):
            if bidirectional:
                if f'rnn_{i}_forward' in self.model_weights:
                    forward_weights = self.model_weights[f'rnn_{i}_forward']
                    self.rnn.forward_cells[i].Wx = forward_weights['Wx']
                    self.rnn.forward_cells[i].Wh = forward_weights['Wh']
                    self.rnn.forward_cells[i].b = forward_weights['b']
                
                if f'rnn_{i}_backward' in self.model_weights:
                    backward_weights = self.model_weights[f'rnn_{i}_backward']
                    self.rnn.backward_cells[i].Wx = backward_weights['Wx']
                    self.rnn.backward_cells[i].Wh = backward_weights['Wh']
                    self.rnn.backward_cells[i].b = backward_weights['b']
            else:
                if f'rnn_{i}' in self.model_weights:
                    rnn_weights = self.model_weights[f'rnn_{i}']
                    self.rnn.forward_cells[i].Wx = rnn_weights['Wx']
                    self.rnn.forward_cells[i].Wh = rnn_weights['Wh']
                    self.rnn.forward_cells[i].b = rnn_weights['b']
        
        self.dropout = DropoutLayer(rate=0.0)
        dense_weights = self.model_weights['dense']
        self.dense = DenseLayer(
            weight=dense_weights['W'],
            bias=dense_weights['b'],
            activation='softmax')
        
    def predict_scratch(self, x):
        if self.embedding is None:
            raise ValueError("From-scratch model not built")
        embedded = self.embedding.forward(x)
        rnn_output = self.rnn.forward(embedded)
        # Take last timestep output
        last_output = rnn_output[:, -1, :]
        dropout_output = self.dropout.forward(last_output)
        predictions = self.dense.forward(dropout_output)
        return predictions
    
    def evaluate_model(self, x_test, y_test, model_type='keras'):
        if model_type == 'keras':
            predictions = self.predict_keras_model(x_test)
        else:
            predictions = self.predict_scratch(x_test)
        
        # Convert predictions to class labels
        y_pred = np.argmax(predictions, axis=1)
        
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"Macro F1-score ({model_type}): {f1_macro:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return f1_macro, y_pred