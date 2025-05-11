import tensorflow as tf
import pickle

def create_and_save_tokenizer(texts, output_dim=128, save_path='src/utils/tokenizer.pkl'):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=None,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        ngrams=None,
        output_mode="int",
        output_sequence_length=None,
        pad_to_max_tokens=False,
        vocabulary=None,
        idf_weights=None,
        sparse=False,
        ragged=False,
        encoding="utf-8",
        name=None
    )
    vectorizer.adapt(texts)

    with open(save_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Tokenizer saved to {save_path}")
    return vectorizer

def load_tokenizer(path='src/utils/tokenizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)
