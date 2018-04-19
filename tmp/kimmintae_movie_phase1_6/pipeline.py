# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import codecs
import re
import string
import os

#===============keras ==============
from keras.preprocessing import text, sequence

#===============morphnizer ============
# from konlpy.tag import Twitter
# twt = Twitter()

class MovieReviewDataset():
    """
    영화리뷰 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str, max_length: int):
        """
        initializer
        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """

        # 데이터, 레이블 각각의 경로
        data_review = os.path.join(dataset_path, 'train', 'train_data')
        data_label = os.path.join(dataset_path, 'train', 'train_label')

        # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
        with open(data_review, 'rt', encoding='utf-8') as f:
            self.reviews = f.readlines()
        # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
        with open(data_label) as f:
            self.labels = [np.float32(x) for x in f.readlines()]
            
        self.dset = pd.DataFrame(data=np.array([self.reviews, self.labels]).T, columns=['reviews', 'labels'])

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.reviews)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.reviews[idx], self.labels[idx]
    
    

    def load_emb_model(self, embedding_path, encodings = "utf-8"):
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        self.emb_model = dict(get_coefs(*o.strip().split(" ")) for o in codecs.open(embedding_path, "r", encodings ))
    
    def dset_regex_morph(self, morph=True):
        docs = []
        for doc in self.dset['reviews']:
            doc = re.sub('[^\w?!]', ' ', doc)
            doc = re.sub('[\s]+', ' ', doc)
            doc = re.sub('[\s]$|^[\s]', '', doc)
#             if morph:
#                 docs.append(" ".join(twt.morphs(doc)))
#             else:
            docs.append(doc)
        self.dset['reviews'] = docs
    
    def dset_embedding(self,
                       emb_model,
                       embed_size = 300,
                       max_features = 100000,
                       maxlen = 20,
                       oov_zero = True,
                       truncating='pre'
                      ):
        
        doc_column = "reviews"
        list_classes = ["labels"]

        list_sentences = self.dset[doc_column].fillna('UNK').values.tolist()
        

        tokenizer = text.Tokenizer(num_words =max_features)
        tokenizer.fit_on_texts(list_sentences)

        list_tokenized = tokenizer.texts_to_sequences(list_sentences)
        
        X = sequence.pad_sequences(list_tokenized, maxlen=maxlen, truncating=truncating)
        Y = self.dset[list_classes].values
        print("=== Data is preprocessed")

        word_index = tokenizer.word_index
        nb_words = min(max_features, len(word_index))

        if oov_zero:
            embedding_matrix = np.zeros((nb_words, embed_size))
        else:
            embedding_matrix = np.random.normal(0.001, 0.4, (nb_words, embed_size))

        for word, i in word_index.items():
            if i >= max_features: continue
            try:
                embedding_vector = emb_model.get(word)
                if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            except: 
                pass
        print("=== Embedding Matrix is loaded")
        
        self.reviews = X
        self.labels = Y
        self.emb_matrix = embedding_matrix
  