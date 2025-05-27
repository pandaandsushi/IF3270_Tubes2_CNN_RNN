import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from utils.layers import EmbeddingLayer, RNNCell, DenseLayer, DropoutLayer
from rnn.rnn_layer import RNNLayer

class RNNModel:
    def __init__(self):
        self.embedding = None
        self.rnn = None
        self.dropout = None
        self.dense = None

    