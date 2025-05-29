import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from utils.data_loader import DataLoader
from utils.tokenizer import TextPreprocessor
from utils.metrics import f1_score_macro, print_classification_report, plot_history
import train_lstm
import random

# Set seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

print("Starting LSTM experiments...")

# --- Global configuration ---
DATASET    = "NusaX"
MAX_LENGTH = train_lstm.MAX_LENGTH
BATCH_SIZE = train_lstm.BATCH_SIZE
EPOCHS     = train_lstm.EPOCHS

# --- Hyperparameter grid ---
param_grid = [
    {"layers": 1, "units": [64],              "bidirectional": False},
    {"layers": 2, "units": [64, 32],          "bidirectional": False},
    {"layers": 2, "units": [32, 16],          "bidirectional": False},
    {"layers": 3, "units": [64, 32, 16],      "bidirectional": False},

    {"layers": 2, "units": [32, 32],          "bidirectional": False},
    {"layers": 2, "units": [64, 64],          "bidirectional": False},
    {"layers": 2, "units": [128, 64],         "bidirectional": False},

    {"layers": 1, "units": [64],              "bidirectional": True},
    {"layers": 2, "units": [32, 16],          "bidirectional": True}
]

# --- Directories setup ---
save_dir = os.path.join("src", "lstm", "experiments")
os.makedirs(save_dir, exist_ok=True)

# --- Load and preprocess data ---
print("Loading NusaX dataset...")
loader = DataLoader(DATASET)
x_tr_raw, x_val_raw, x_te_raw, y_tr, y_val, y_te = loader.load_data()

x_tr_txt = x_tr_raw[:, 0].astype(str)
x_val_txt = x_val_raw[:, 0].astype(str)
x_te_txt = x_te_raw[:, 0].astype(str)

if y_tr.dtype == 'object' or y_tr.dtype.kind in {'U', 'S'}:
    print("Converting string labels to numeric...")
    label_encoder = LabelEncoder()
    y_tr = label_encoder.fit_transform(y_tr)
    y_val = label_encoder.transform(y_val)
    y_te = label_encoder.transform(y_te)

y_tr = y_tr.astype(np.int32)
y_val = y_val.astype(np.int32)
y_te = y_te.astype(np.int32)

print(f"Data loaded successfully!")
print(f"Label range: {np.min(y_tr)} to {np.max(y_tr)}")
print(f"Number of classes: {len(np.unique(y_tr))}")

tp = TextPreprocessor(max_features=5000, max_length=MAX_LENGTH)
tp.create_text_vectorization(x_tr_txt)
X_train = tp.preprocess_with_keras(x_tr_txt)
X_val = tp.preprocess_with_keras(x_val_txt)
X_test = tp.preprocess_with_keras(x_te_txt)
vocab_size = tp.get_vocab_size()
num_classes = len(np.unique(y_tr))

print(f"Vocabulary size: {vocab_size}")
print(f"Training data shape: {X_train.shape}")

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_tr),
    y=y_tr
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# --- Experiment loop ---
results = []
for experiment_idx, cfg in enumerate(param_grid):
    # reset seeds for each experiment
    experiment_seed = SEED + experiment_idx
    np.random.seed(experiment_seed)
    tf.random.set_seed(experiment_seed)
    random.seed(experiment_seed)
    
    layers = cfg["layers"]
    units = cfg["units"]
    bidi_flag = cfg["bidirectional"]
    tag = f"L{layers}_U{'-'.join(map(str,units))}_B{int(bidi_flag)}"
    print(f"\n=== Running experiment: {tag} (seed: {experiment_seed}) ===")

    # Override hyperparams in train_lstm
    train_lstm.LSTM_LAYERS = layers
    train_lstm.LSTM_UNITS = units
    train_lstm.BIDIRECTIONAL = bidi_flag
    train_lstm.SEED = experiment_seed

    # Build & train model
    model = train_lstm.build_lstm_model(vocab_size, num_classes)
    
    callbacks = [
        train_lstm.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        train_lstm.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2
    )

    model_path = os.path.join(save_dir, f"model_{tag}.keras")
    model.save(model_path)
    print(f"Model saved: {model_path}")

    # Save training curves
    curve_path = os.path.join(save_dir, f"history_{tag}.png")
    plot_history(history, save_path=curve_path)
    print(f"Training curves saved to {curve_path}")

    # Evaluate on test set
    loss, acc = model.evaluate(X_test, y_te, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
    f1 = f1_score_macro(y_te, y_pred)
    print(f"Result {tag}: loss={loss:.4f}, acc={acc:.4f}, macro-F1={f1:.4f}")

    # Classification report
    print(f"\n[Classification Report for {tag}]")
    print_classification_report(y_te, y_pred)
    
    # Save classification report to file
    report_txt = classification_report(y_te, y_pred, zero_division=0)
    cr_path = os.path.join(save_dir, f"class_report_{tag}.txt")
    with open(cr_path, "w") as f:
        f.write(f"Experiment Configuration: {tag}\n")
        f.write(f"Seed: {experiment_seed}\n")
        f.write(f"Layers: {layers}\n")
        f.write(f"Units: {units}\n")
        f.write(f"Bidirectional: {bidi_flag}\n")
        f.write(f"Epochs trained: {len(history.history['loss'])}\n")
        f.write(f"Final Results:\n")
        f.write(f"Loss: {loss:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_txt)
    print(f"Classification report saved to {cr_path}")

    # Collect results
    results.append({
        "config": tag,
        "seed": experiment_seed,
        "layers": layers,
        "units": "-".join(map(str, units)),
        "bidirectional": int(bidi_flag),
        "epochs_trained": len(history.history['loss']),
        "loss": float(loss),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "model_path": model_path,
        "curve_path": curve_path,
        "report_path": cr_path
    })

# --- Save summary to CSV ---
csv_path = os.path.join(save_dir, "results_lstm.csv")
df = pd.DataFrame(results)
df = df.sort_values('macro_f1', ascending=False)
df.to_csv(csv_path, index=False)
print(f"\nAll experiments completed. Summary saved to {csv_path}")
print("\nTop 3 configurations by Macro F1:")
print(df[['config', 'seed', 'accuracy', 'macro_f1']].head(3))
