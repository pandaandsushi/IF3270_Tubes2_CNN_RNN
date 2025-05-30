import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from utils.data_loader import DataLoader
from utils.tokenizer import TextPreprocessor
from utils.metrics import f1_score_macro, print_classification_report, plot_history
import random

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Hyperparameter configuration
DATASET       = "NusaX"
EMBEDDING_DIM = 64
MAX_LENGTH    = 100
LSTM_LAYERS   = 2
LSTM_UNITS    = [32, 16]
BIDIRECTIONAL = False
DROPOUT_RATE  = 0.5
EPOCHS        = 15
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

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
            mask_zero=True,
            name="embedding",
            embeddings_initializer=keras.initializers.RandomUniform(seed=SEED)
        )(inputs)
    
    x = keras.layers.Dropout(0.3, seed=SEED)(x)
    
    # lstm layers
    for i, units in enumerate(LSTM_UNITS):
        return_seq = (i < LSTM_LAYERS - 1)
        if BIDIRECTIONAL:
            x = keras.layers.Bidirectional(
                    keras.layers.LSTM(
                        units, 
                        return_sequences=return_seq,
                        dropout=0.3,
                        recurrent_dropout=0.3,
                        kernel_regularizer=keras.regularizers.l2(0.01),
                        kernel_initializer=keras.initializers.GlorotUniform(seed=SEED),
                        recurrent_initializer=keras.initializers.Orthogonal(seed=SEED)
                    ),
                    name=f"bidi_lstm_{i+1}"
                )(x)
        else:
            x = keras.layers.LSTM(
                    units, 
                    return_sequences=return_seq,
                    dropout=0.3,
                    recurrent_dropout=0.3,
                    kernel_regularizer=keras.regularizers.l2(0.01),
                    name=f"lstm_{i+1}",
                    kernel_initializer=keras.initializers.GlorotUniform(seed=SEED),
                    recurrent_initializer=keras.initializers.Orthogonal(seed=SEED)
                )(x)
        
        x = keras.layers.Dropout(0.4, seed=SEED)(x)
    
    x = keras.layers.Dense(32, activation='relu', 
                          kernel_regularizer=keras.regularizers.l2(0.01),
                          kernel_initializer=keras.initializers.GlorotUniform(seed=SEED))(x)
    x = keras.layers.Dropout(DROPOUT_RATE, seed=SEED)(x)
    
    # output layer
    outputs = keras.layers.Dense(
                  num_classes,
                  activation="softmax",
                  name="output_softmax",
                  kernel_initializer=keras.initializers.GlorotUniform(seed=SEED)
              )(x)
    
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model = keras.Model(inputs, outputs, name="LSTM_Classifier")
    model.compile(
        optimizer=optimizer,
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
    x_tr_txt = x_tr_raw[:, 1].astype(str)
    x_val_txt= x_val_raw[:, 1].astype(str)
    x_te_txt = x_te_raw[:, 1].astype(str)

    # convert labels to numeric if they are strings
    if y_tr.dtype == 'object' or y_tr.dtype.kind in {'U', 'S'}:
        print("Converting string labels to numeric...")
        label_encoder = LabelEncoder()
        y_tr = label_encoder.fit_transform(y_tr)
        y_val = label_encoder.transform(y_val)
        y_te = label_encoder.transform(y_te)
    
    y_tr = y_tr.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_te = y_te.astype(np.int32)

    print(f"Label range: {np.min(y_tr)} to {np.max(y_tr)}")
    print(f"Number of classes: {len(np.unique(y_tr))}")
    
    # check class distribution
    unique, counts = np.unique(y_tr, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_tr),
        y=y_tr
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)

    tp = TextPreprocessor(max_features=5000, max_length=MAX_LENGTH)
    tp.create_text_vectorization(x_tr_txt)
    x_tr = tp.preprocess_with_keras(x_tr_txt)
    x_val = tp.preprocess_with_keras(x_val_txt)
    x_te  = tp.preprocess_with_keras(x_te_txt)
    vocab_size = tp.get_vocab_size()

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training data shape: {x_tr.shape}")

    model = build_lstm_model(vocab_size, num_classes=len(np.unique(y_tr)))
    
    print("\nModel summary:")
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
        validation_freq=1
    )

    plot_history(history, save_path="src/lstm/lstm_training_history.png")
    print("Training curves saved to src/lstm/lstm_training_history.png")

    os.makedirs("src/lstm", exist_ok=True)
    model.save("src/lstm/lstm_model.keras")
    print("Keras LSTM model saved at src/lstm/lstm_model.keras")

    loss, acc = model.evaluate(x_te, y_te, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(model.predict(x_te, batch_size=BATCH_SIZE, verbose=0), axis=1)
    f1 = f1_score_macro(y_te, y_pred)
    
    print(f"\n[Keras LSTM] Test Results:")
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")

    print("\n[Keras LSTM] Classification Report:")
    print_classification_report(y_te, y_pred)

    report_txt = classification_report(y_te, y_pred, zero_division=0)
    with open("src/lstm/classification_report_keras_lstm.txt", "w") as f:
        f.write(f"Model Configuration:\n")
        f.write(f"EMBEDDING_DIM: {EMBEDDING_DIM}\n")
        f.write(f"MAX_LENGTH: {MAX_LENGTH}\n")
        f.write(f"LSTM_UNITS: {LSTM_UNITS}\n")
        f.write(f"BIDIRECTIONAL: {BIDIRECTIONAL}\n")
        f.write(f"DROPOUT_RATE: {DROPOUT_RATE}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"EPOCHS_TRAINED: {len(history.history['loss'])}\n")
        f.write(f"VOCAB_SIZE: {vocab_size}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"Test Loss: {loss:.4f}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1-Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_txt)
    print("Classification report saved to src/lstm/classification_report_keras_lstm.txt")

if __name__ == "__main__":
    main()