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
        if self.keras_model is None:
            raise ValueError("No trained model to save")
        
        self.keras_model.save_weights(filepath)  
        
        weights = {}
        layer_idx = 0
        
        print("Extracting weights from Keras model...")
        
        for layer in self.keras_model.layers:
            if isinstance(layer, layers.Embedding):
                weights['embedding'] = layer.get_weights()[0]
                print(f"Extracted embedding weights: {weights['embedding'].shape}")
                
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
                    
                    print(f"Extracted bidirectional RNN layer {layer_idx} weights:")
                    print(f"  Forward - Wx: {forward_weights[0].shape}, Wh: {forward_weights[1].shape}, b: {forward_weights[2].shape}")
                    print(f"  Backward - Wx: {backward_weights[0].shape}, Wh: {backward_weights[1].shape}, b: {backward_weights[2].shape}")
                    
                else:
                    # unidirectional RNN
                    layer_weights = layer.get_weights()
                    weights[f'rnn_{layer_idx}'] = {
                        'Wx': layer_weights[0],
                        'Wh': layer_weights[1],
                        'b': layer_weights[2]
                    }
                    
                    print(f"Extracted unidirectional RNN layer {layer_idx} weights:")
                    print(f"  Wx: {layer_weights[0].shape}, Wh: {layer_weights[1].shape}, b: {layer_weights[2].shape}")
                
                layer_idx += 1
                
            elif isinstance(layer, layers.Dense):
                layer_weights = layer.get_weights()
                weights['dense'] = {
                    'W': layer_weights[0],
                    'b': layer_weights[1]
                }
                print(f"Extracted dense weights: W: {layer_weights[0].shape}, b: {layer_weights[1].shape}")
        
        self.model_weights = weights
        print("Weight extraction completed!")

    def rnn_scratch(self, rnn_layers, rnn_units, bidirectional=False):
        if not self.model_weights:
            raise ValueError("No weights available. Train a model first.")
        
        print("Building from-scratch model...")
        
        # Initialize embedding layer
        self.embedding = EmbeddingLayer(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            weights=self.model_weights['embedding']
        )
        print(f"Embedding layer initialized with shape: {self.model_weights['embedding'].shape}")
        
        # Initialize RNN layer
        self.rnn = RNNLayer(
            input_dim=self.embedding_dim,
            hidden_dims=rnn_units,
            bidirectional=bidirectional
        )
        print(f"RNN layer initialized: {rnn_layers} layers, units: {rnn_units}, bidirectional: {bidirectional}")
        
        # Set RNN weights from trained model
        for i in range(rnn_layers):
            print(f"Setting weights for RNN layer {i}...")
            
            if bidirectional:
                # Check if forward weights exist
                if f'rnn_{i}_forward' in self.model_weights:
                    forward_weights = self.model_weights[f'rnn_{i}_forward']
                    self.rnn.forward_cells[i].Wx = forward_weights['Wx']
                    self.rnn.forward_cells[i].Wh = forward_weights['Wh']
                    self.rnn.forward_cells[i].b = forward_weights['b']
                    print(f"  Forward weights set for layer {i}")
                else:
                    print(f"  Warning: Forward weights not found for layer {i}")
                
                # Check if backward weights exist
                if f'rnn_{i}_backward' in self.model_weights:
                    backward_weights = self.model_weights[f'rnn_{i}_backward']
                    if i < len(self.rnn.backward_cells):
                        self.rnn.backward_cells[i].Wx = backward_weights['Wx']
                        self.rnn.backward_cells[i].Wh = backward_weights['Wh']
                        self.rnn.backward_cells[i].b = backward_weights['b']
                        print(f"  Backward weights set for layer {i}")
                    else:
                        print(f"  Error: Backward cell {i} not found in RNN layer")
                else:
                    print(f"  Warning: Backward weights not found for layer {i}")
            else:
                # Unidirectional RNN
                if f'rnn_{i}' in self.model_weights:
                    rnn_weights = self.model_weights[f'rnn_{i}']
                    self.rnn.forward_cells[i].Wx = rnn_weights['Wx']
                    self.rnn.forward_cells[i].Wh = rnn_weights['Wh']
                    self.rnn.forward_cells[i].b = rnn_weights['b']
                    print(f"  Unidirectional weights set for layer {i}")
                else:
                    print(f"  Warning: Weights not found for layer {i}")
        
        # Initialize dropout layer (no dropout during inference)
        self.dropout = DropoutLayer(rate=0.0)
        
        # Initialize dense layer
        dense_weights = self.model_weights['dense']
        self.dense = DenseLayer(
            weight=dense_weights['W'],
            bias=dense_weights['b'],
            activation='softmax'
        )
        print(f"Dense layer initialized with shape: W: {dense_weights['W'].shape}, b: {dense_weights['b'].shape}")
        
        print("From-scratch model built successfully!")
        
    def predict_scratch(self, x):
        if self.embedding is None:
            raise ValueError("From-scratch model not built")
        
        print(f"Running from-scratch prediction on {x.shape[0]} samples...")
        
        # Embedding
        embedded = self.embedding.forward(x)
        print(f"After embedding: {embedded.shape}")
        
        # RNN
        rnn_output = self.rnn.forward(embedded)
        print(f"After RNN: {rnn_output.shape}")
        
        # Take last timestep output
        last_output = rnn_output[:, -1, :]
        print(f"After taking last timestep: {last_output.shape}")
        
        # Dropout (no effect during inference)
        dropout_output = self.dropout.forward(last_output)
        print(f"After dropout: {dropout_output.shape}")
        
        # Dense layer
        predictions = self.dense.forward(dropout_output)
        print(f"Final predictions: {predictions.shape}")
        
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
