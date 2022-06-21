from typing import List

import tensorflow as tf
import os
import app


def create_model(encoder: tf.keras.layers.TextVectorization, vocab_list_fp: str) -> tf.keras.Model:
    '''
    V1 06.18.2022,
    Post-creation
    :param encoder: Adapted TextVectorization layer or None
    :param vocab_list_fp: Full path for processed list of vocab
    :return: model
    '''
    VOCAB_SIZE = 1500
    input = tf.keras.Input((1,), dtype=tf.dtypes.string)
    if not encoder:
        encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        # Adapt a vocab list from the training data
        with open(vocab_list_fp, "r+") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            encoder.adapt(lines)

    layer = encoder(input)
    layer = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE,
                                      output_dim=64,
                                      mask_zero=True)(layer)
    layer = tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(units=64))(layer)
    layer = tf.keras.layers.Dense(64, activation="relu")(layer)
    layer = tf.keras.layers.Dense(1)(layer)

    return tf.keras.Model(inputs=input, outputs=layer, name="nlp_rnn_classifier")

def load_data(fn:str, fp=None):
    '''
    V1 06.18.2022,
    Post-creation
    :param fn: local file name
    :param fp: full path for dataset
    :return: map of { text : emotion }
    '''
    if fp:
        txt_path = fp
    else:
        txt_path = os.path.join(app.root(), "datasets", "emotions", fn)

    res = {}
    with open(txt_path) as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split(";")
            line, emotion = line_split
            emotion = emotion.strip()
            if emotion == 'joy' or emotion == 'anger':
                res[line] = emotion.strip()
    print(f'loaded {len(res)} items')
    return res

def load_data_externally(fp: str):
    return load_data(None, fp)