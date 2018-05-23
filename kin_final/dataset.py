
# -*- coding: utf-8 -*-

import os
import numpy as np
from kor_char_parser import decompose_str_as_one_hot

class KinQueryDataset:
    def __init__(self, dataset_path: str, max_length: int):

        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')

        with open(queries_path, 'rt', encoding='utf8') as f:
            self.queries = f.readlines()

        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]


def get_tknzr(data: list):
    from konlpy.tag import Twitter
    from keras.preprocessing.text import Tokenizer
    t = Twitter()
    queries_noun = [(' '.join(t.nouns(i.split('\t')[0]))+ '\t' + ' '.join(t.nouns(i.split('\t')[1])) + '\n').replace('내공', '') for i in data]
    tknzr = Tokenizer(num_words=None, filters='"#$&()*+,-./:;<=>@[\]_`{|}~', lower=True, char_level=True)
    tknzr.fit_on_texts(data)
    return tknzr

def preprocess(data: list, tknzr, maxlen: int):
    from konlpy.tag import Twitter
    from keras.preprocessing.sequence import pad_sequences
    t = Twitter()
    queries_noun = [(' '.join(t.nouns(i.split('\t')[0]))+ '\t' + ' '.join(t.nouns(i.split('\t')[1])) + '\n').replace('내공', '') for i in data]
    vectorized_data = tknzr.texts_to_sequences(data)
    padded_data = pad_sequences(vectorized_data, maxlen=maxlen, truncating='post')
    return padded_data