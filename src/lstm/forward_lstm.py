import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from utils.layers import EmbeddingLayer, LSTMCell, DenseLayer, DropoutLayer
from utils.metrics import f1_score_macro, print_classification_report
from utils.data_loader import DataLoader
from utils.tokenizer import TextPreprocessor
import random

# Set seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Configuration
DATASET       = "NusaX"
MAX_LENGTH    = 100
EMBEDDING_DIM = 64
LSTM_LAYERS   = 2
LSTM_UNITS    = [32, 16]
BIDIRECTIONAL = False

def load_keras_weights(model_path: str):
    """
    Load a Keras model and return all its weights as a list.
    """
    model = tf.keras.models.load_model(model_path)
    return model

def inspect_model_architecture(keras_model):
    """
    Inspect the actual architecture of the loaded model.
    """
    print("\n=== Model Architecture Inspection ===")
    for i, layer in enumerate(keras_model.layers):
        print(f"Layer {i}: {layer.name} - {type(layer).__name__}")
        if hasattr(layer, 'output_shape'):
            print(f"  Output shape: {layer.output_shape}")
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()
            print(f"  Weights shapes: {[w.shape for w in weights]}")
    print("=" * 50)

def build_scratch_layers(keras_model):
    """
    Extract custom layers from a Keras model and return them as scratch layers.
    """
    inspect_model_architecture(keras_model)
    
    emb_weights = keras_model.get_layer("embedding").get_weights()[0]
    emb = EmbeddingLayer(input_dim=emb_weights.shape[0],
                         output_dim=emb_weights.shape[1],
                         weights=emb_weights)
    emb.embedding_matrix[0, :] = 0.0
    print(f"Embedding: {emb_weights.shape}")

    lstm_cells = []
    lstm_layers = []
    
    for layer in keras_model.layers:
        if 'lstm' in layer.name.lower() or 'bidirectional' in layer.name.lower():
            lstm_layers.append(layer)
    
    print(f"Found {len(lstm_layers)} LSTM layers:")
    
    for i, layer in enumerate(lstm_layers):
        print(f"Processing layer: {layer.name}")
        weights = layer.get_weights()
        
        if len(weights) == 3:
            W, U, b = weights
            # LSTM gates
            Wi, Wf, Wc, Wo = np.split(W, 4, axis=1)
            Ui, Uf, Uc, Uo = np.split(U, 4, axis=1)
            bi, bf, bc, bo = np.split(b, 4)
            
            print(f"  LSTM {i}: Input {W.shape[0]} -> Output {W.shape[1]//4}")
            lstm_cells.append(LSTMCell(Wi, Wf, Wo, Wc, Ui, Uf, Uo, Uc, bi, bf, bo, bc))
            
        elif len(weights) == 6:
            W_f, U_f, b_f, W_b, U_b, b_b = weights
            
            # forward LSTM
            Wi_f, Wf_f, Wc_f, Wo_f = np.split(W_f, 4, axis=1)
            Ui_f, Uf_f, Uc_f, Uo_f = np.split(U_f, 4, axis=1)
            bi_f, bf_f, bc_f, bo_f = np.split(b_f, 4)
            
            # backward LSTM
            Wi_b, Wf_b, Wc_b, Wo_b = np.split(W_b, 4, axis=1)
            Ui_b, Uf_b, Uc_b, Uo_b = np.split(U_b, 4, axis=1)
            bi_b, bf_b, bc_b, bo_b = np.split(b_b, 4)
            
            print(f"  Bidirectional LSTM {i}: Input {W_f.shape[0]} -> Output {W_f.shape[1]//4 * 2}")
            
            lstm_cells.append({
                'forward': LSTMCell(Wi_f, Wf_f, Wo_f, Wc_f, Ui_f, Uf_f, Uo_f, Uc_f, bi_f, bf_f, bo_f, bc_f),
                'backward': LSTMCell(Wi_b, Wf_b, Wo_b, Wc_b, Ui_b, Uf_b, Uo_b, Uc_b, bi_b, bf_b, bo_b, bc_b),
                'is_bidirectional': True
            })
        else:
            raise ValueError(f"Unexpected number of weights in LSTM layer: {len(weights)}")

    intermediate_dense_layer = keras_model.get_layer("dense")
    Wd_int, bd_int = intermediate_dense_layer.get_weights()
    intermediate_dense = DenseLayer(weight=Wd_int, bias=bd_int, activation="relu")
    print(f"Intermediate Dense: {Wd_int.shape}")

    dense_layer = keras_model.get_layer("output_softmax")
    Wd, bd = dense_layer.get_weights()
    dense = DenseLayer(weight=Wd, bias=bd, activation="softmax")
    print(f"Output Dense: {Wd.shape}")

    return emb, lstm_cells, intermediate_dense, dense

def scratch_forward(emb, lstm_cells, intermediate_dense, dense, X):
    """
    Perform forward propagation through the custom LSTM model.
    """
    np.random.seed(SEED)
    
    N, seq_len = X.shape
    H = emb.forward(X)
    
    mask = (X != 0)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sample (first sequence): {mask[0]}")
    
    dropout_layer = DropoutLayer(rate=0.3)
    dropout_layer.setis_training(False)
    current_output = dropout_layer.forward(H)
    
    for i, cell in enumerate(lstm_cells):
        if isinstance(cell, dict) and cell.get('is_bidirectional', False):
            forward_cell = cell['forward']
            backward_cell = cell['backward']
            
            x_sample = current_output[:, 0, :]
            h_sample_f, c_sample_f = forward_cell.forward(x_sample, 
                                                         np.zeros((N, x_sample.shape[1])), 
                                                         np.zeros((N, x_sample.shape[1])))
            
            h_f = np.zeros((N, h_sample_f.shape[1]))
            c_f = np.zeros((N, c_sample_f.shape[1]))
            h_b = np.zeros((N, h_sample_f.shape[1]))
            c_b = np.zeros((N, c_sample_f.shape[1]))
            
            forward_outputs = []
            backward_outputs = []
            
            # forward pass
            for t in range(seq_len):
                x_t = current_output[:, t, :]
                
                active_mask = mask[:, t]
                if active_mask.any():
                    active_indices = np.where(active_mask)[0]
                    
                    if len(active_indices) == N:
                        h_f, c_f = forward_cell.forward(x_t, h_f, c_f)
                    else:
                        h_active, c_active = forward_cell.forward(
                            x_t[active_indices], 
                            h_f[active_indices], 
                            c_f[active_indices]
                        )
                        h_f[active_indices] = h_active
                        c_f[active_indices] = c_active
                
                forward_outputs.append(h_f.copy())
            
            # backward pass
            for t in range(seq_len-1, -1, -1):
                x_t = current_output[:, t, :]
                
                active_mask = mask[:, t]
                if active_mask.any():
                    active_indices = np.where(active_mask)[0]
                    
                    if len(active_indices) == N:
                        h_b, c_b = backward_cell.forward(x_t, h_b, c_b)
                    else:
                        h_active, c_active = backward_cell.forward(
                            x_t[active_indices], 
                            h_b[active_indices], 
                            c_b[active_indices]
                        )
                        h_b[active_indices] = h_active
                        c_b[active_indices] = c_active
                
                backward_outputs.append(h_b.copy())
            
            backward_outputs = backward_outputs[::-1]
            
            if i == len(lstm_cells) - 1:
                current_output = np.concatenate([forward_outputs[-1], backward_outputs[-1]], axis=1)
            else:
                bidirectional_outputs = []
                for t in range(seq_len):
                    concat_output = np.concatenate([forward_outputs[t], backward_outputs[t]], axis=1)
                    bidirectional_outputs.append(concat_output)
                current_output = np.stack(bidirectional_outputs, axis=1)
        
        else:
            # unidirectional LSTM
            print(f"Processing LSTM layer {i}")
            print(f"  Input shape: {current_output.shape}")
            
            lstm_hidden_size = cell.Ui.shape[0]
            print(f"  LSTM hidden size: {lstm_hidden_size}")
            
            h = np.zeros((N, lstm_hidden_size))
            c = np.zeros((N, lstm_hidden_size))
            
            outputs = []
            for t in range(seq_len):
                x_t = current_output[:, t, :]
                
                active_mask = mask[:, t]
                if active_mask.any():
                    active_indices = np.where(active_mask)[0]
                    
                    if len(active_indices) == N:
                        h, c = cell.forward(x_t, h, c)
                    else:
                        h_active, c_active = cell.forward(
                            x_t[active_indices], 
                            h[active_indices], 
                            c[active_indices]
                        )
                        h[active_indices] = h_active
                        c[active_indices] = c_active
                
                outputs.append(h.copy())
            
            if i == len(lstm_cells) - 1:
                current_output = outputs[-1]
                print(f"  Final output shape: {current_output.shape}")
            else:
                current_output = np.stack(outputs, axis=1)
                print(f"  Intermediate output shape: {current_output.shape}")
        
        dropout_layer = DropoutLayer(rate=0.4)
        dropout_layer.setis_training(False)
        if i == len(lstm_cells) - 1:
            current_output = dropout_layer.forward(current_output)
        else:
            pass
    
    dropout_layer = DropoutLayer(rate=0.5)
    dropout_layer.setis_training(False)
    current_output = dropout_layer.forward(current_output)
    
    print(f"Before intermediate dense layer: {current_output.shape}")
    current_output = intermediate_dense.forward(current_output)
    print(f"After intermediate dense layer: {current_output.shape}")
    
    print(f"Before output dense layer: {current_output.shape}")
    y_prob = dense.forward(current_output)
    print(f"After output dense layer: {y_prob.shape}")
    
    # temperature = 1.1
    # y_prob = y_prob ** (1/temperature)
    # y_prob = y_prob / np.sum(y_prob, axis=1, keepdims=True)
    
    # class_boost = np.array([1.05, 1.0, 1.0])
    # y_prob = y_prob * class_boost
    # y_prob = y_prob / np.sum(y_prob, axis=1, keepdims=True)
    
    return y_prob

def main():
    # load dataset
    print(f"[Scratch] Loading dataset: {DATASET}...")
    loader = DataLoader(DATASET)
    x_tr_raw, x_val_raw, x_te_raw, y_tr, y_val, y_te = loader.load_data()

    x_tr_txt = x_tr_raw[:, 1].astype(str)
    x_te_txt = x_te_raw[:, 1].astype(str)

    # label conversion
    if y_te.dtype == 'object' or y_te.dtype.kind in {'U', 'S'}:
        print("Converting string labels to numeric...")
        label_encoder = LabelEncoder()
        if y_tr.dtype == 'object' or y_tr.dtype.kind in {'U', 'S'}:
            y_tr = label_encoder.fit_transform(y_tr)
            y_te = label_encoder.transform(y_te)
        else:
            y_te = label_encoder.fit_transform(y_te)
    
    y_te = y_te.astype(np.int32)

    # text preprocessing
    tp = TextPreprocessor(max_features=5000, max_length=100)
    tp.create_text_vectorization(x_tr_txt)
    x_te = tp.preprocess_with_keras(x_te_txt)

    print(f"Test data shape: {x_te.shape}")

    # load and inspect model
    print("Loading Keras model...")
    keras_model = load_keras_weights("src/lstm/lstm_model.keras")
    
    emb, lstm_cells, intermediate_dense, dense = build_scratch_layers(keras_model)

    print("Performing scratch forward propagation...")
    y_prob = scratch_forward(emb, lstm_cells, intermediate_dense, dense, x_te)
    y_pred = np.argmax(y_prob, axis=1)

    f1 = f1_score_macro(y_te, y_pred)
    print(f"\n[Scratch LSTM] macro-F1: {f1:.4f}")

    print("\n[Scratch LSTM] Classification Report:")
    print_classification_report(y_te, y_pred)

    report_txt = classification_report(y_te, y_pred, zero_division=0)
    with open("src/lstm/classification_report_scratch_lstm.txt", "w") as f:
        f.write(f"Scratch LSTM Forward Propagation Results:\n")
        f.write(f"MAX_LENGTH: {MAX_LENGTH}\n")
        f.write(f"EMBEDDING_DIM: {EMBEDDING_DIM}\n")
        f.write(f"LSTM_UNITS: {LSTM_UNITS}\n")
        f.write(f"BIDIRECTIONAL: {BIDIRECTIONAL}\n\n")
        f.write(f"Macro F1-Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_txt)
    print("Classification report saved to src/lstm/classification_report_scratch_lstm.txt")

if __name__ == "__main__":
    main()