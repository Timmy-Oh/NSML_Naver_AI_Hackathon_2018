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

from keras.callbacks import Callback
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess



def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    TIME_STEPS = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    #a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = merge([inputs, a_probs], mode='mul')
    return output_attention_mul



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
        pred = model.predict(preprocessed_data)[-1]
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
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--cell_size', type=int, default=60)
    args.add_argument('--cell_size2', type=int, default=40)
    args.add_argument('--embed_size', type=int, default=300)
    args.add_argument('--prob_dropout', type=float, default=0.5)
    args.add_argument('--max_features', type=int, default=251)
    args.add_argument('--batch_size', type=int, default=64)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = 'C:/Users/dhzns/tmp/kimmintae_kin_phase2_17/data/kin/'

    def get_model(config):
        inp = Input(shape=(config.strmaxlen, ), name='input')
        emb = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
        emb1 = SpatialDropout1D(config.prob_dropout)(emb)
        x1 = Bidirectional(CuDNNLSTM(config.cell_size, return_sequences=True))(emb1)
        x12 = Bidirectional(CuDNNGRU(config.cell_size, return_sequences=True))(x1)
        x12c = Conv1D(64, kernel_size = 2, strides=1, padding = "valid", kernel_initializer = "he_uniform")(x12)
        
        avg_pool1 = GlobalAveragePooling1D()(x1)
        max_pool1 = GlobalMaxPooling1D()(x1)
        
        avg_pool12 = GlobalAveragePooling1D()(x12)
        max_pool12 = GlobalMaxPooling1D()(x12)
        
        avg_pool12c = GlobalAveragePooling1D()(x12c)
        max_pool12c = GlobalMaxPooling1D()(x12c)
        
        conc = concatenate([avg_pool1, max_pool1, avg_pool12, max_pool12, avg_pool12c, max_pool12c])
#         fc1 = Dense(50, activation='relu')(conc)
        fc1 = Dropout(config.prob_dropout)(conc)
        rnnc_out = Dense(2,activation='softmax')(fc1)
        
        emb2 = SpatialDropout1D(config.prob_dropout)(emb)
        r1 = Bidirectional(CuDNNLSTM(config.cell_size2, return_sequences=True))(emb2)
        r12 = Bidirectional(CuDNNLSTM(config.cell_size2, return_sequences=False))(r1)
      
        rfc1 = Dense(50, activation='relu')(r12)
        rfc1 = Dropout(config.prob_dropout)(rfc1)
        rnn_out = Dense(2,activation='softmax')(rfc1)
        
        
        stack_layer = concatenate([rnn_out, rnnc_out])
        ens_out = Dense(2, activation='softmax', use_bias=False)(stack_layer)

        model = Model(inputs=inp, outputs=[rnn_out, rnnc_out, ens_out])
        model.compile(loss='categorical_crossentropy', optimizer='adam', loss_weights=[1., 0.7, 0.2], metrics=['categorical_crossentropy', 'accuracy'])
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
        hist = model.fit(x, [y_onehot,y_onehot,y_onehot], batch_size=config.batch_size, callbacks=[nsml_callback], epochs=config.epochs,
        # validation_split=0.1, 
        verbose=2)
        

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)