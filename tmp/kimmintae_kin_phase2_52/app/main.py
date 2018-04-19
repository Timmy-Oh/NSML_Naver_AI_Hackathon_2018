
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


import argparse
import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
# keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, add, multiply, Lambda, merge
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, Conv1D
from keras.layers import Dropout, SpatialDropout1D, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, PReLU
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling1D
from keras.layers import K, Activation
from keras.engine import Layer
from keras.models import load_model
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.utils import to_categorical
from keras.layers import TimeDistributed
from keras.callbacks import Callback
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        model.save_weights(filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        model.load_weights(filename)
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
#         pred = model.predict(preprocessed_data)[-1]
        pred = model.predict(preprocessed_data)
        pred_prob = pred[:,1]
        clipped = np.argmax(pred, axis=-1)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        
        return list(zip(pred_prob.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다
    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
class Nsml_Callback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs, step=epoch)
        nsml.save(epoch)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--embed_size', type=int, default=256)
    args.add_argument('--prob_dropout', type=float, default=0.5)
    args.add_argument('--max_features', type=int, default=251)
    args.add_argument('--batch_size', type=int, default=512)
    args.add_argument('--learning_rate', type=float, default=0.0008)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../data/kin'

#     def get_model(config):
#         inp = Input(shape=(config.strmaxlen, ), name='input')
#         emb = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
#         drop = Dropout(config.prob_dropout)(emb)
#         x1 = Conv1D(1024, kernel_size = 3, padding="valid", activation='relu', kernel_initializer = "he_uniform")(drop)
#         avg_pool = GlobalAveragePooling1D()(x1)
#         x1 =Dense(256, activation='relu')(avg_pool)
#         x1 =Dense(128, activation='relu')(x1)
#         x1 =Dense(64, activation='relu')(x1)
#         avg_out = Dense(2,activation='softmax')(x1)
        
#         x2 = Conv1D(1024, kernel_size = 3, padding="valid", activation='relu', kernel_initializer = "he_uniform")(drop)
#         max_pool = GlobalMaxPooling1D()(x2)
#         x2 =Dense(256, activation='relu')(max_pool)
#         x2 =Dense(128, activation='relu')(x2)
#         x2 =Dense(64, activation='relu')(x2)
#         max_out = Dense(2,activation='softmax')(x2)
        
#         stack = concatenate([avg_out, max_out])
#         out = Dense(2,activation='softmax')(stack)

#         model = Model(inputs=inp, outputs=[avg_out, max_out, out])
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy'])
#         return model
    def get_model(config):
        def td_avg(x):
            return K.mean(x, axis=1)
        inp = Input(shape=(config.strmaxlen, ), name='input')
        emb = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
        drop = SpatialDropout1D(config.prob_dropout)(emb)
        x1 = Conv1D(512, kernel_size = 3, padding="valid", activation='relu', kernel_initializer = "he_uniform")(drop)
        x2 = Conv1D(512, kernel_size = 5, padding="valid", activation='relu', kernel_initializer = "he_uniform")(drop)
        x3 = Conv1D(512, kernel_size = 7, padding="valid", activation='relu', kernel_initializer = "he_uniform")(drop)
        x1_avg = GlobalAveragePooling1D()(x1)
        x2_avg = GlobalAveragePooling1D()(x2)
        x3_avg = GlobalAveragePooling1D()(x3)
        x1_max = GlobalMaxPooling1D()(x1)
        x2_max = GlobalMaxPooling1D()(x2)
        x3_max = GlobalMaxPooling1D()(x3)
        
        x = concatenate([x1_avg, x2_avg, x3_avg, x1_max, x2_max, x3_max])
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(config.prob_dropout)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(config.prob_dropout)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(config.prob_dropout)(x)
        out = Dense(2,activation='softmax')(x)
       
        model = Model(inputs=inp, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(config.learning_rate), metrics=['categorical_crossentropy', 'accuracy'])
        return model

    model = get_model(config)

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        # 데이터를 로드합니다.
        # epoch마다 학습을 수행합니다.
        nsml_callback = Nsml_Callback()
#         print(dataset.labels)
        y = np.array(dataset.labels)
        x = np.array(dataset.queries)
        y_onehot = to_categorical(y, num_classes=2)
#         X_train, X_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.05, random_state=55)
#         hist = model.fit(x, [y_onehot,y_onehot,y_onehot], batch_size=config.batch_size, callbacks=[nsml_callback], epochs=config.epochs, validation_split=0.1, verbose=1)
        hist = model.fit(x, y_onehot, batch_size=config.batch_size, callbacks=[nsml_callback], epochs=config.epochs, verbose=2)
#         hist = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=config.batch_size, callbacks=[nsml_callback], epochs=config.epochs, verbose=2)
        

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/sample'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)