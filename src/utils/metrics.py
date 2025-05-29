import numpy as np
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

def f1_score_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average='macro')

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names=None):
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print(report)

def plot_history(history, save_path=None):
    # loss
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],   marker='o', label='train loss')
    plt.plot(history.history['val_loss'], marker='x', label='val loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # accuracy
    if 'accuracy' in history.history:
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'],    marker='o', label='train acc')
        plt.plot(history.history['val_accuracy'],marker='x', label='val acc')
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
