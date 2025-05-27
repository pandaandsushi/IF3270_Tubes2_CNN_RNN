import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

class TextPreprocessor:
    def __init__(self, max_features=10000, max_length=100):
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = None
        self.text_vectorization = None
        
    def create_text_vectorization(self, texts):
        # Create TextVectorization layer for tokenization
        self.text_vectorization = tf.keras.layers.TextVectorization(
            max_tokens=self.max_features,
            output_sequence_length=self.max_length,
            output_mode='int'
        )
        
        # Adapt to the texts
        self.text_vectorization.adapt(texts)
        
        return self.text_vectorization
    
    def preprocess_with_keras(self, texts):
        # Preprocess texts using Keras TextVectorization
        if self.text_vectorization is None:
            self.create_text_vectorization(texts)
        
        # Convert texts to vector sequences
        sequences = self.text_vectorization(texts)
        
        return sequences.numpy()
    
    def get_vocab_size(self):
        if self.text_vectorization is not None:
            return self.text_vectorization.vocabulary_size()
        elif self.tokenizer is not None:
            return len(self.tokenizer.word_index) + 1
        else:
            return self.max_features
