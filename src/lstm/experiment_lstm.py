import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from utils.data_loader import DataLoader
from utils.tokenizer import TextPreprocessor
from utils.metrics import f1_score_macro, print_classification_report, plot_history
import train_lstm

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
    {"layers": 3, "units": [64, 32, 16],      "bidirectional": False},

    {"layers": 2, "units": [32, 32],          "bidirectional": False},
    {"layers": 2, "units": [64, 64],          "bidirectional": False},
    {"layers": 2, "units": [128, 128],        "bidirectional": False},

    {"layers": 2, "units": [64, 32],          "bidirectional": False},
    {"layers": 2, "units": [64, 32],          "bidirectional": True}
]

# --- Directories setup ---
save_dir = os.path.join("src", "lstm", "experiments")
os.makedirs(save_dir, exist_ok=True)

# --- Load and preprocess data ---
print("Loading NusaX dataset...")
loader = DataLoader(DATASET)
xt_raw, xv_raw, xt_raw, yt, yv, y_test = loader.load_data()

# Text preprocessing
print(f"Data loaded successfully!")
print(f"Preprocessing data...")
texts_train = xt_raw[:, 0].astype(str)
tp = TextPreprocessor(max_features=10000, max_length=MAX_LENGTH)
tp.create_text_vectorization(texts_train)
X_train = tp.preprocess_with_keras(texts_train)
X_val   = tp.preprocess_with_keras(xv_raw[:, 0].astype(str))
X_test  = tp.preprocess_with_keras(xt_raw[:, 0].astype(str))
vocab_size  = tp.get_vocab_size()
num_classes = len(np.unique(yt))

# --- Experiment loop ---
results = []
for cfg in param_grid:
    layers    = cfg["layers"]
    units     = cfg["units"]
    bidi_flag = cfg["bidirectional"]
    tag = f"L{layers}_U{'-'.join(map(str,units))}_B{int(bidi_flag)}"
    print(f"\n=== Running experiment: {tag} ===")

    # Override hyperparams in train_lstm
    train_lstm.LSTM_LAYERS   = layers
    train_lstm.LSTM_UNITS    = units
    train_lstm.BIDIRECTIONAL = bidi_flag

    # Build & train model
    model = train_lstm.build_lstm_model(vocab_size, num_classes)
    history = model.fit(
        X_train, yt,
        validation_data=(X_val, yv),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
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
    loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred    = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE), axis=1)
    f1        = f1_score_macro(y_test, y_pred)
    print(f"Result {tag}: loss={loss:.4f}, acc={acc:.4f}, macro-F1={f1:.4f}")

    # Classification report
    print(f"\n[Classification Report for {tag}]")
    print_classification_report(y_test, y_pred)
    # Save classification report to file
    report_txt = classification_report(y_test, y_pred, zero_division=0)
    cr_path = os.path.join(save_dir, f"class_report_{tag}.txt")
    with open(cr_path, "w") as f:
        f.write(report_txt)
    print(f"Classification report saved to {cr_path}")

    # Collect results
    results.append({
        "config": tag,
        "layers": layers,
        "units": "-".join(map(str,units)),
        "bidirectional": int(bidi_flag),
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
df.to_csv(csv_path, index=False)
print(f"\nAll experiments completed. Summary saved to {csv_path}")
