import os
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report
from utils.data_loader import DataLoader
from utils.tokenizer import TextPreprocessor
from utils.metrics import f1_score_macro, print_classification_report, plot_history

# Hyperparameter configuration
DATASET       = "NusaX"
EMBEDDING_DIM = 100
MAX_LENGTH    = 100
LSTM_LAYERS   = 2
LSTM_UNITS    = [64, 32]
BIDIRECTIONAL = True
DROPOUT_RATE  = 0.2
EPOCHS        = 10
BATCH_SIZE    = 32

def build_lstm_model(vocab_size, num_classes):
    """
    Builds a Keras model with the following architecture:
    Embedding → (Bi)LSTM → Dropout → Dense (softmax).
    """
    inputs = keras.Input(shape=(MAX_LENGTH,), name="input_text")
    
    # embedding layer
    x = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LENGTH,
            name="embedding"
        )(inputs)
    
    # lstm layers
    for i, units in enumerate(LSTM_UNITS):
        return_seq = (i < LSTM_LAYERS - 1)
        lstm_cls   = keras.layers.LSTM
        layer_name = f"lstm_{i+1}"
        if BIDIRECTIONAL:
            x = keras.layers.Bidirectional(
                    lstm_cls(units, return_sequences=return_seq),
                    name=f"bidi_{layer_name}"
                )(x)
        else:
            x = lstm_cls(units, return_sequences=return_seq, name=layer_name)(x)
    
    # dropout layer
    x = keras.layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    
    # output layer
    outputs = keras.layers.Dense(
                  num_classes,
                  activation="softmax",
                  name="output_softmax"
              )(x)
    
    model = keras.Model(inputs, outputs, name="LSTM_Classifier")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    # load dataset
    print(f"Loading dataset: {DATASET}...")
    loader = DataLoader(DATASET)
    x_tr_raw, x_val_raw, x_te_raw, y_tr, y_val, y_te = loader.load_data()

    # extract text data
    x_tr_txt = x_tr_raw[:, 0].astype(str)
    x_val_txt= x_val_raw[:, 0].astype(str)
    x_te_txt = x_te_raw[:, 0].astype(str)

    # preprocess text
    tp = TextPreprocessor(max_features=10000, max_length=MAX_LENGTH)
    tp.create_text_vectorization(x_tr_txt)
    x_tr = tp.preprocess_with_keras(x_tr_txt)
    x_val = tp.preprocess_with_keras(x_val_txt)
    x_te  = tp.preprocess_with_keras(x_te_txt)
    vocab_size = tp.get_vocab_size()

    model = build_lstm_model(vocab_size, num_classes=len(np.unique(y_tr)))
    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    plot_history(history, save_path="src/lstm/lstm_training_history.png")
    print("Training curves saved to src/lstm/lstm_training_history.png")

    os.makedirs("src/lstm", exist_ok=True)
    model.save("src/lstm/lstm_model.keras")
    print("Keras LSTM model saved at src/lstm/lstm_model.keras")

    loss, acc = model.evaluate(x_te, y_te, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(model.predict(x_te, batch_size=BATCH_SIZE), axis=1)
    f1 = f1_score_macro(y_te, y_pred)
    print(f"[Keras LSTM] Test loss: {loss:.4f}, acc: {acc:.4f}, macro-F1: {f1:.4f}")

    print("\n[Keras LSTM] Classification Report:")
    print_classification_report(y_te, y_pred)

    report_txt = classification_report(y_te, y_pred, zero_division=0)
    with open("src/lstm/classification_report_keras_lstm.txt", "w") as f:
        f.write(report_txt)
    print("Classification report saved to src/lstm/classification_report_keras_lstm.txt")

if __name__ == "__main__":
    main()