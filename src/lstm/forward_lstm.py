import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from utils.layers import EmbeddingLayer, LSTMCell, DenseLayer
from utils.metrics import f1_score_macro, print_classification_report
from utils.data_loader import DataLoader
from utils.tokenizer import TextPreprocessor

# Configuration
DATASET       = "NusaX"
MAX_LENGTH    = 100
EMBEDDING_DIM = 100
LSTM_LAYERS   = 2
LSTM_UNITS    = [64, 32]
BIDIRECTIONAL = True  # bi and uni

def load_keras_weights(model_path: str):
    """
    Load a Keras model and return all its weights as a list.
    """
    model = tf.keras.models.load_model(model_path)
    return model

def build_scratch_layers(keras_model):
    """
    Extract custom layers from a Keras model and return them as scratch layers.
    """
    emb_weights = keras_model.get_layer("embedding").get_weights()[0]
    emb = EmbeddingLayer(input_dim=emb_weights.shape[0],
                         output_dim=emb_weights.shape[1],
                         weights=emb_weights)

    lstm_cells = []
    for i in range(LSTM_LAYERS):
        prefix = f"bidi_lstm_{i+1}" if BIDIRECTIONAL else f"lstm_{i+1}"
        layer = keras_model.get_layer(prefix)
        W, U, b = layer.get_weights()

        units = LSTM_UNITS[i]
        Wi, Wf, Wc, Wo = np.split(W, 4, axis=1)
        Ui, Uf, Uc, Uo = np.split(U, 4, axis=1)
        bi, bf, bc, bo = np.split(b, 4)

        lstm_cells.append(LSTMCell(Wi, Wf, Wo, Wc, Ui, Uf, Uo, Uc, bi, bf, bo, bc))

    dense_layer = keras_model.get_layer("output_softmax")
    Wd, bd = dense_layer.get_weights()
    dense = DenseLayer(weight=Wd, bias=bd, activation="softmax")

    return emb, lstm_cells, dense

def scratch_forward(emb, lstm_cells, dense, X):
    """
    Perform forward propagation through the custom LSTM model.
    """
    N = X.shape[0]
    H = emb.forward(X)

    h = [np.zeros((N, units)) for units in LSTM_UNITS]
    c = [np.zeros((N, units)) for units in LSTM_UNITS]

    for t in range(MAX_LENGTH):
        x_t = H[:, t, :]
        for i, cell in enumerate(lstm_cells):
            h_prev, c_prev = h[i], c[i]
            h_i, c_i = cell.forward(x_t, h_prev, c_prev)
            x_t = h_i
            h[i], c[i] = h_i, c_i

    final_h = h[-1]
    y_prob = dense.forward(final_h)
    return y_prob

def main():
    # load dataset
    print(f"[Scratch] Loading dataset: {DATASET}...")
    loader = DataLoader(DATASET)
    x_tr, x_val, x_te, y_tr, y_val, y_te = loader.load_data()

    tp = TextPreprocessor(max_features=10000, max_length=MAX_LENGTH)
    tp.create_text_vectorization(x_tr[:, 0].astype(str))
    X_test = tp.preprocess_with_keras(x_te[:, 0].astype(str))

    # load model
    keras_model = load_keras_weights("src/lstm/lstm_model.keras")
    emb, lstm_cells, dense = build_scratch_layers(keras_model)

    y_prob = scratch_forward(emb, lstm_cells, dense, X_test)
    y_pred = np.argmax(y_prob, axis=1)

    f1 = f1_score_macro(y_te, y_pred)
    print(f"[Scratch LSTM] macro-F1: {f1:.4f}")

    print("\n[Scratch LSTM] Classification Report:")
    print_classification_report(y_te, y_pred)

    report_txt = classification_report(y_te, y_pred, zero_division=0)
    with open("src/lstm/classification_report_scratch_lstm.txt", "w") as f:
        f.write(report_txt)
    print("Classification report saved to src/lstm/classification_report_scratch_lstm.txt")

if __name__ == "__main__":
    main()