# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import tensorflow as tf

### custom
import pandas as pd
import numpy as np
import codecs
import re
import string
import os

# keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, add, multiply
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, Conv1D
from keras.layers import Dropout, SpatialDropout1D, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, PReLU
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling1D
from keras.layers import K, Activation
from keras.engine import Layer

from keras.callbacks import Callback

### NSML
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from pipeline import MovieReviewDataset

# IS_ON_NSML = False
# HAS_DATASET = False

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard

def bind_model(model):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        # directory
        model.save(filename=filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        model.load_weights(filename)
    
    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = MovieReviewDataset(raw_data, 400)
#         model.evaluate()

        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model.predict(preprocessed_data)
        point = output_prediction.data.squeeze(dim=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
     
    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)
    
class Nsml_Callback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        bind_model(model=self.model)
    
if __name__ == '__main__':

    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=5)
    args.add_argument('--strmaxlen', type=int, default=400)
    
    args.add_argument('--maxlen', type=int, default=20)
    args.add_argument('--cell_size', type=int, default=40)
    args.add_argument('--embed_size', type=int, default=300)
    args.add_argument('--prob_dropout', type=float, default=0.4)
    args.add_argument('--max_features', type=int, default=431)
    
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    
    config = args.parse_args()


    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen
    output_size = 1
    hidden_layer_size = 200
    learning_rate = 0.001
    character_size = 251
    emb_train = False

    def get_model(embedding_matrix, config):
        inp = Input(shape=(config.maxlen, ), name='input')
        x1 = Embedding(config.max_features, config.embed_size, weights=[embedding_matrix], trainable = emb_train)(inp)
        x1 = SpatialDropout1D(config.prob_dropout)(x1)
#         if cell_type_GRU:
        x1 = Bidirectional(CuDNNLSTM(config.cell_size, return_sequences=True))(x1)
        x1 = Bidirectional(CuDNNGRU(config.cell_size, return_sequences=True))(x1)
#         else :
#             x1 = Bidirectional(CuDNNLSTM(config.cell_size, return_sequences=True))(x1)
#             x1 = Bidirectional(CuDNNLSTM(config.cell_size, return_sequences=True))(x1)
        avg_pool1 = GlobalAveragePooling1D()(x1)
        max_pool1 = GlobalMaxPooling1D()(x1)
        ##merge
        conc = concatenate([avg_pool1, max_pool1])
        outp = Dense(output_size)(conc)
        # DONOTCHANGE: Reserved for nsml
        
        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])
        
        return model
    
        

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())
    
    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        
        ### csutom
        dataset.dset_regex_morph(morph=True)
        emb_path="./wiki.ko.vec"
        dataset.load_emb_model(embedding_path=emb_path)
        dataset.dset_embedding(dataset.emb_model)
        
        mdl = get_model(dataset.emb_matrix, config)
        nsml_callback = Nsml_Callback()
        hist = mdl.fit(dataset.reviews, dataset.labels, batch_size=config.batch, epochs=config.epochs, callbacks = [nsml_callback], verbose=2)
#             nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
#                         train__loss=float(avg_loss/one_batch_size), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
#             nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
            
        res = nsml.infer(queries)
        print(res)