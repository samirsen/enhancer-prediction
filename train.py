import numpy as np
import tensorflow as tf
import random

def separate_data(data, labels):
    pos_examples = data[labels == 1]
    neg_examples = data[labels == -1]
    return pos_examples, neg_examples

def split_data(pos_data, neg_data, SPLIT=0.8):
    indices = list(range(len(pos_data)))
    random.shuffle(indices)
    pos_train = indices[:int(len(pos_data) * SPLIT), :]
    pos_valid = indices[int(len(pos_data) * SPLIT:), :]

    indices = list(range(len(neg_data)))
    random.shuffle(indices)
    neg_train = indices[:int(len(neg_data) * SPLIT), :]
    neg_valid = indices[int(len(neg_data) * SPLIT):, :]

    return pos_train, neg_train, pos_valid, neg_valid
