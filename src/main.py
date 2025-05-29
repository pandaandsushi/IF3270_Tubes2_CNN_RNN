# main_fixed.py

from utils.data_loader import DataLoader
from utils.tokenizer import TextPreprocessor
from rnn.rnn_model import RNNModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DATASET = "NusaX"
EMBEDDING_DIM = 100
MAX_LENGTH = 100
RNN_LAYERS = 2
RNN_UNITS = [64, 32]
BIDIRECTIONAL = True
DROPOUT_RATE = 0.2
EPOCHS = 5
BATCH_SIZE = 32

print("Loading NusaX dataset...")
data_loader = DataLoader(DATASET)
x_train_raw, x_val_raw, x_test_raw, y_train_raw, y_val_raw, y_test_raw = data_loader.load_data()

print(f"Data loaded successfully!")
print(f"Train set: {x_train_raw.shape}, {y_train_raw.shape}")
print(f"Validation set: {x_val_raw.shape}, {y_val_raw.shape}")
print(f"Test set: {x_test_raw.shape}, {y_test_raw.shape}")

# --- Extract text data ---
x_train_texts = x_train_raw[:, 1].astype(str)
x_val_texts = x_val_raw[:, 1].astype(str)
x_test_texts = x_test_raw[:, 1].astype(str)

# Print some examples to verify
print("\nExample texts from training set:")
for i in range(min(3, len(x_train_texts))):
    print(f"Text {i+1}: {x_train_texts[i]}")
    print(f"Label: {y_train_raw[i]}")
    print(f"X TRAIN RAWWWWW: {x_train_raw[i]}")
    print("-" * 50)

# --- Handle string labels ---
print("\nProcessing labels...")
print(f"Unique labels in training set: {np.unique(y_train_raw)}")

# Create label encoder to convert string labels to integers
label_encoder = LabelEncoder()

# Fit the encoder on all labels (train + val + test) to ensure consistency
all_labels = np.concatenate([y_train_raw, y_val_raw, y_test_raw])
label_encoder.fit(all_labels)

# Transform labels to integers
y_train = label_encoder.transform(y_train_raw)
y_val = label_encoder.transform(y_val_raw)
y_test = label_encoder.transform(y_test_raw)

# Print label mapping
print("\nLabel mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label} -> {i}")

# Get number of classes
NUM_CLASSES = len(label_encoder.classes_)
print(f"Number of unique classes: {NUM_CLASSES}")

# Print label distribution
print(f"\nLabel distribution in training set:")
unique_labels, counts = np.unique(y_train, return_counts=True)
for label_idx, count in zip(unique_labels, counts):
    label_name = label_encoder.classes_[label_idx]
    print(f"  {label_name} ({label_idx}): {count} samples ({count/len(y_train)*100:.1f}%)")

# --- Preprocess Text ---
print("\nPreprocessing text data...")
tokenizer = TextPreprocessor(max_features=10000, max_length=MAX_LENGTH)
tokenizer.create_text_vectorization(x_train_texts)

x_train = tokenizer.preprocess_with_keras(x_train_texts)
x_val = tokenizer.preprocess_with_keras(x_val_texts)
x_test = tokenizer.preprocess_with_keras(x_test_texts)
vocab_size = tokenizer.get_vocab_size()

print(f"Vocabulary size: {vocab_size}")
print(f"Processed train shape: {x_train.shape}")
print(f"Processed validation shape: {x_val.shape}")
print(f"Processed test shape: {x_test.shape}")

# Print example
print("\nExample of processed text (first 20 tokens):")
# print(f"Original: {x_train_texts[0]}")
# print(f"Processed: {x_train[0][:50]}")
print(f"Original: {x_train_texts[:5]}")
print(f"Processed: {x_train[:5]}")

# --- Model Training ---
print("\nInitializing and training model...")
model = RNNModel(vocab_size, EMBEDDING_DIM, MAX_LENGTH, NUM_CLASSES)
history = model.train_keras_model(
    x_train, y_train,
    x_val, y_val,
    rnn_layers=RNN_LAYERS,
    rnn_units=RNN_UNITS,
    bidirectional=BIDIRECTIONAL,
    dropout_rate=DROPOUT_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# --- Extract Weights and Predict with Scratch RNN ---
print("\nExtracting weights and building from-scratch model...")
model.save_model_weights('keras_weights.weights.h5')
model.rnn_scratch(rnn_layers=RNN_LAYERS, rnn_units=RNN_UNITS, bidirectional=BIDIRECTIONAL)

# --- Evaluation ---
print("\nEvaluating Keras model...")
f1_keras, y_pred_keras = model.evaluate_model(x_test, y_test, model_type='keras')

print("\nEvaluating from-scratch model...")
f1_scratch, y_pred_scratch = model.evaluate_model(x_test, y_test, model_type='scratch')

# --- Additional Analysis ---
print("\nAdditional Analysis:")
print(f"Keras model F1-score: {f1_keras:.4f}")
print(f"From-scratch model F1-score: {f1_scratch:.4f}")
print(f"Difference: {abs(f1_keras - f1_scratch):.4f}")

# Check prediction agreement
match_percentage = np.mean(y_pred_keras == y_pred_scratch) * 100
print(f"Prediction match percentage: {match_percentage:.2f}%")

# --- Plotting ---
plt.figure(figsize=(16, 5))

# Training and Validation Loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# Training and Validation Accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title("Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# F1 Scores
plt.subplot(1, 3, 3)
plt.bar(['Keras Model', 'Scratch RNN'], [f1_keras, f1_scratch], color=['blue', 'orange'])
plt.title("Macro F1 Score Comparison")
plt.ylabel("F1 Score")
plt.grid(True, alpha=0.3, axis='y')

for i, v in enumerate([f1_keras, f1_scratch]):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_performance1.png', dpi=300, bbox_inches='tight')
plt.show()

print("Results saved to 'model_performance.png'")

# Save label encoder for future use
# import pickle
# with open('label_encoder.pkl', 'wb') as f:
#     pickle.dump(label_encoder, f)
# print("Label encoder saved to 'label_encoder.pkl'")
