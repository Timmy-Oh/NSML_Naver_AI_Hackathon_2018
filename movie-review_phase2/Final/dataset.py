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
            loaded = f.readlines()
            self.reviews_pre = preprocess_pre(loaded, max_length)
#             self.reviews_post = preprocess_post(loaded, max_length)
        # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
        with open(data_label) as f:
            self.labels = [np.float32(x) for x in f.readlines()]

def preprocess_pre(data: list, max_length: int):
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    def text_norm(sent):
        sent = re.sub('[\,\<\>\(\)\+\-\=\&\@\#\$]', '', sent)
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

# def preprocess_post(data: list, max_length: int):
#     """
#      입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
#      기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
#      문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

#     :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
#     :param max_length: 문자열의 최대 길이
#     :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
#     """
#     def text_norm(sent):
#         sent = re.sub('[\,\<\>\(\)\+\-\=\&\@\#\$]', '', sent)
#         sent = re.sub('\.{2,}', '..', sent)
#         sent = re.sub('\~+', '~', sent)
#         sent = re.sub('\!+', '!', sent)
#         sent = re.sub('\?+', '?', sent)
#         sent = re.sub('ㅋ{1,}|ㅎ{1,}', 'ㅋ', sent)
#         sent = re.sub('ㅜ{1,}|ㅠ{1,}|ㅠㅜ|ㅜㅠ\ㅡㅜ\ㅜㅡ\ㅡㅠ\ㅠㅡ', 'ㅠㅠ', sent)
#         return sent
    
#     vectorized_data = [decompose_str_as_one_hot(text_norm(datum), warning=False) for datum in data]
#     zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
#     for idx, seq in enumerate(vectorized_data):
#         length = len(seq)
#         if length >= max_length:
#             length = max_length
#             zero_padding[idx, :length] = np.array(seq)[:length]
#         else:
#             zero_padding[idx, :length] = np.array(seq)
#     return zero_padding