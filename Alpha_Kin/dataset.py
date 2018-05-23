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

from kor_char_parser import decompose_str_as_one_hot
from konlpy.tag import Twitter

class KinDataset():
    def __init__(self, dataset_path: str, max_length: int):
        # 데이터, 레이블 각각의 경로
        data_review = os.path.join(dataset_path, 'train', 'train_data')
        data_label = os.path.join(dataset_path, 'train', 'train_label')
        twt = Twitter()

        # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
        with open(data_review, 'rt', encoding='utf-8') as f:
            loaded = f.readlines()
            nouns = [" ".join(twt.nouns(doc)) for doc in loaded]
            self.reviews = preprocess(nouns, max_length)
            
        # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
        with open(data_label) as f:
            self.labels = [np.float32(x) for x in f.readlines()]

def preprocess(data: list, max_length: int):
    def text_norm(sent):
        sent = re.sub('\.{2,}', '..', sent)
        sent = re.sub('\~+', '~', sent)
        sent = re.sub('\!+', '!', sent)
        sent = re.sub('\?+', '?', sent)
        sent = re.sub('ㅋ{1,}|ㅎ{1,}', 'ㅋ', sent)
        sent = re.sub('ㅜ{1,}|ㅠ{1,}|ㅠㅜ|ㅜㅠ\ㅡㅜ\ㅜㅡ\ㅡㅠ\ㅠㅡ', 'ㅠㅠ', sent)
        return sent
    
    vectorized_data = [decompose_str_as_one_hot(datum, warning=False) for datum in data]
    zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
    for idx, seq in enumerate(vectorized_data):
        length = len(seq)
        if length >= max_length:
            length = max_length
            zero_padding[idx, (max_length-length):] = np.array(seq)[:length]
        else:
            zero_padding[idx, (max_length-length):] = np.array(seq)
    return zero_padding