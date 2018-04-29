# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import re

import numpy as np

from scipy.sparse import hstack
from konlpy.tag import Twitter

class MovieReviewDataset():
    """
    영화리뷰 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str):
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


def regexp(texts):
    twt = Twitter()
    container = []
    for i, sent in enumerate(texts):
        if i % 200000 == 0:
            print(i)
        sent = re.sub('[\,\<\>\(\)\+\-\=\&\@\#\$]', '', sent)
        sent = re.sub('\.{2,}', ' .. ', sent)
        sent = re.sub('\~+', ' ~ ', sent)
        sent = re.sub('\!+', ' ! ', sent)
        sent = re.sub('\?+', ' ? ', sent)
        sent = re.sub('(ac)', ' 99', sent)
        sent = re.sub('(mv)', ' 88', sent)
        sent = re.sub('ㅋ{1,}|ㅎ{1,}', 'ㅋ', sent)
        sent = re.sub('ㅜ{1,}|ㅠ{1,}|ㅠㅜ|ㅜㅠ\ㅡㅜ\ㅜㅡ\ㅡㅠ\ㅠㅡ', 'ㅠㅠ', sent)
        container.append(" ".join(twt.morphs(sent)))
    return container

def word_preprocessor(sent):
    twt = Twitter()
    sent = re.sub('[\,\<\>\(\)\+\-\=\&\@\#\$]', '', sent)
    sent = re.sub('\.{2,}', ' .. ', sent)
    sent = re.sub('\~+', ' ~ ', sent)
    sent = re.sub('\!+', ' ! ', sent)
    sent = re.sub('\?+', ' ? ', sent)
    sent = re.sub('(ac)', ' 99', sent)
    sent = re.sub('(mv)', ' 88', sent)
    sent = re.sub('ㅋ{1,}|ㅎ{1,}', 'ㅋ', sent)
    sent = re.sub('ㅜ{1,}|ㅠ{1,}|ㅠㅜ|ㅜㅠ\ㅡㅜ\ㅜㅡ\ㅡㅠ\ㅠㅡ', 'ㅠㅠ', sent)
    sent = " ".join(twt.morphs(sent))
    return sent

def char_preprocessor(sent):
    sent = re.sub('[\,\<\>\(\)\+\-\=\&\@\#\$]', '', sent)
    sent = re.sub('\.{2,}', '..', sent)
    sent = re.sub('\~+', '~', sent)
    sent = re.sub('\!+', '!', sent)
    sent = re.sub('\?+', '?', sent)
    sent = re.sub('(ac)', '', sent)
    sent = re.sub('(mv)', '', sent)
    sent = re.sub('ㅋ{1,}|ㅎ{1,}', 'ㅋ', sent)
    sent = re.sub('ㅜ{1,}|ㅠ{1,}|ㅠㅜ|ㅜㅠ\ㅡㅜ\ㅜㅡ\ㅡㅠ\ㅠㅡ', 'ㅠㅠ', sent)
    sent = re.sub('[1234567890]', '', sent)
    return sent

# 144570
def trn_val_seperation(dataset: list, bound: int):
    bound = -1 * bound
    X_trn = dataset.reviews[:bound]
    X_val = dataset.reviews[bound:]
    Y_trn = dataset.labels[:bound]
    Y_val = dataset.labels[bound:]

    return X_trn, X_val, Y_trn, Y_val
        
def vect_fit(review, vect_word, vect_char):
    vect_word.fit(review)
    vect_char.fit(review)
    return vect_word, vect_char

def vect_transform(review, vect_word, vect_char):
    df_word = vect_word.transform(review)
    df_char = vect_char.transform(review)
    return hstack([df_word, df_char]).tocsr()