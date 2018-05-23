# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np

# from sklearn.model_selection import KFold
# keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, add, multiply, average
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, Conv1D, Dropout
from keras.layers import Dropout, SpatialDropout1D, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, PReLU
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling1D
from keras.layers import K, Activation
from keras.engine import Layer

##DPCNN
from keras.models import Model
from keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, SpatialDropout1D
from keras.layers import add, Dropout, PReLU, BatchNormalization, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import optimizers
from keras import initializers, regularizers, constraints, callbacks

from keras.callbacks import Callback
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint

import nsml
from dataset import KinDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.utils import to_categorical


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
        nouns = [" ".join(twt.nouns(raw_data)) for doc in loaded]
        preprocessed_data = preprocess(nouns, config.strmaxlen)
        pred = model.predict(preprocessed_data)[-1]
        pred_prob = pred[:,1]
        clipped = np.argmax(pred, axis=-1)
        return list(zip(pred_prob.flatten(), clipped.flatten()))

    nsml.bind(save=save, load=load, infer=infer)

class Nsml_Callback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs, step=epoch)
        nsml.save(epoch)

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers import LSTM, activations, Wrapper, Recurrent
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--strmaxlen', type=int, default=240)
    
#   # args.add_argument('--maxlen', type=int, default=200)
    args.add_argument('--cell_size_l1', type=int, default=50)
    args.add_argument('--cell_size_l2', type=int, default=40)
    args.add_argument('--filter_size', type=int, default=32)
    args.add_argument('--kernel_size', type=int, default=3)
    args.add_argument('--embed_size', type=int, default=256)
    args.add_argument('--prob_dropout', type=float, default=0.4)
    args.add_argument('--prob_dropout2', type=float, default=0.2)
    args.add_argument('--max_features', type=int, default=251)
    args.add_argument('--batch_size', type=int, default=100)
    
    config = args.parse_args()
    
    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = './sample_data/kin'
        
    def get_model(config):
        inp = Input(shape=(config.strmaxlen, ), name='input')
#         inp = Input(shape=(config.max_features, ), name='input')

        emb = Embedding(config.max_features, config.max_features,embeddings_initializer='identity', trainable = True)(inp)
#         emb1 = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
        emb1 = SpatialDropout1D(config.prob_dropout)(emb)
        
        #### 
        l1_L = Bidirectional(CuDNNLSTM(config.cell_size_l1, return_sequences=True))(emb1)
        
        l2_LL = Bidirectional(CuDNNLSTM(config.cell_size_l2, return_sequences=False))(l1_L)
        l2_LG = Bidirectional(CuDNNGRU(config.cell_size_l2, return_sequences=False))(l1_L)
        
        out_LL = Dropout(config.prob_dropout2)(l2_LL)
        out_LG = Dropout(config.prob_dropout2)(l2_LG)
        
        out_LL = Dense(2, activation='softmax')(out_LL)
        out_LG = Dense(2, activation='softmax')(out_LG)
        ####
        
        emb2 = SpatialDropout1D(config.prob_dropout)(emb)
        
        l1_G = Bidirectional(CuDNNGRU(config.cell_size_l1, return_sequences=True))(emb2)
        
        l2_GL = Bidirectional(CuDNNLSTM(config.cell_size_l2, return_sequences=False))(l1_G)
        l2_GG = Bidirectional(CuDNNGRU(config.cell_size_l2, return_sequences=False))(l1_G)
        
        out_GL = Dropout(config.prob_dropout2)(l2_GL)
        out_GG = Dropout(config.prob_dropout2)(l2_GG)
        out_GL = Dense(2, activation='softmax')(out_GL)
        out_GG = Dense(2, activation='softmax')(out_GG)
        
        out_avg = average([out_LL, out_LG, out_GL, out_GG])

        
# #         ==================================================================================================
        model_avg = Model(inputs=inp, outputs=[out_LL, out_LG, out_GL, out_GG, out_avg])
        
        model_avg.compile(loss='categorical_crossentropy', optimizer='adam',
                          loss_weights=[1., 1., 1., 1., 0.1],
                          metrics=['accuracy'])
        
        return model_avg
    
    print("model creating...")
    model = get_model(config)
    model.summary()
    
    # DONOTCHANGE: Reserved for nsml use
    print("nsml binding...")
    bind_model(model, config)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        print("data loading...")
        dataset = KinDataset(DATASET_PATH, config.strmaxlen)
        
        x_pre = np.array(dataset.reviews)
        y = np.array(dataset.labels)
        y = to_categorical(y, num_classes=2)
        
        from sklearn.utils import shuffle
        x_pre, y = shuffle(x_pre, y, random_state=1991)
        
        # epoch마다 학습을 수행합니다.
        nsml_callback = Nsml_Callback()
        print("model training...")
        hist = model.fit(x_pre, [y,y,y,y,y], 
#                          validation_split = 0.12,
                         batch_size=config.batch_size, callbacks=[nsml_callback], epochs=config.epochs, verbose=2)

    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'test/test_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)