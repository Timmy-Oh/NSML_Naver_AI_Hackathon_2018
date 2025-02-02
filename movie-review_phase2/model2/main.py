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
from dataset import MovieReviewDataset, preprocess_pre, preprocess_post
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML


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
        preprocessed_data = preprocess_pre(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        point = model.predict(preprocessed_data)[-1]
        point[point>10.] = 10.
        point[point<1.] = 1.

        point = point.squeeze(axis=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

class Nsml_Callback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs, step=epoch)
        nsml.save(epoch)

from keras import backend as K
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
    args.add_argument('--strmaxlen', type=int, default=200)
    
#   # args.add_argument('--maxlen', type=int, default=200)
    args.add_argument('--cell_size_l1', type=int, default=40)
    args.add_argument('--cell_size_l2', type=int, default=40)
    args.add_argument('--filter_size', type=int, default=32)
    args.add_argument('--kernel_size', type=int, default=2)
    args.add_argument('--embed_size', type=int, default=256)
    args.add_argument('--prob_dropout', type=float, default=0.4)
    args.add_argument('--max_features', type=int, default=251)
    args.add_argument('--batch_size', type=int, default=256)
    
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../data/movie_review/'
        
    def get_model(config):
        inp = Input(shape=(config.strmaxlen, ), name='input')
        
        emb1 = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
        emb1 = SpatialDropout1D(config.prob_dropout)(emb1)
        
        #### 
        l1_G = Bidirectional(CuDNNGRU(config.cell_size_l1, return_sequences=True))(emb1)
        
        l2_LL = Bidirectional(CuDNNLSTM(config.cell_size_l2, return_sequences=True))(l1_G)
        l2_LG = Bidirectional(CuDNNGRU(config.cell_size_l2, return_sequences=True))(l1_G)

#         avg_pool_L = GlobalAveragePooling1D()(l1_L)
#         max_pool_L = GlobalMaxPooling1D()(l1_L)

        attention_LL = Attention(config.strmaxlen)(l2_LL)
        attention_LG = Attention(config.strmaxlen)(l2_LG)    
#         avg_pool_LL = GlobalAveragePooling1D()(l2_LL)
#         max_pool_LL = GlobalMaxPooling1D()(l2_LL)
#         avg_pool_LG = GlobalAveragePooling1D()(l2_LG)
#         max_pool_LG = GlobalMaxPooling1D()(l2_LG)

#         avg_pool_LLC = GlobalAveragePooling1D()(l3_LLC)
#         max_pool_LLC = GlobalMaxPooling1D()(l3_LLC)
#         avg_pool_LGC = GlobalAveragePooling1D()(l3_LGC)
#         max_pool_LGC = GlobalMaxPooling1D()(l3_LGC)
        
#         conc_LL = concatenate([avg_pool_LL, max_pool_LL])
#         conc_LG = concatenate([avg_pool_LG, max_pool_LG])
#         conc_GLC = concatenate([avg_pool_G, max_pool_G, avg_pool_GL, max_pool_GL, avg_pool_GLC, max_pool_GLC])
#         conc_GGC = concatenate([avg_pool_G, max_pool_G, avg_pool_GG, max_pool_GG, avg_pool_GGC, max_pool_GGC])        

        out_LL = Dropout(config.prob_dropout)(attention_LL)
        out_LL = Dense(1)(out_LL)
        out_LG = Dropout(config.prob_dropout)(attention_LG)
        out_LG = Dense(1)(out_LG)

        
        ####
        
        out_avg = average([out_LL, out_LG])

        
# #         ==================================================================================================
        model_avg = Model(inputs=inp, outputs=[out_LL, out_LG, out_avg])
        
#         inp_pre = Input(shape=(config.strmaxlen, ), name='input_pre')
#         inp_post = Input(shape=(config.strmaxlen, ), name='input_post')
        
#         model_pre = model_avg(inp_pre)
#         model_post = model_avg(inp_post)
        
#         stack_layer = concatenate([model_pre, model_post])
#         ens_out = Dense(1, use_bias=False)(stack_layer)
        
#         reg_model = Model(inputs=[inp_pre, inp_post], outputs=ens_out)
        
        model_avg.compile(loss='mean_squared_error', optimizer='adam', loss_weights=[1., 1., 4.] ,metrics=['mean_squared_error', 'accuracy'])
        
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
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        
        x_pre = np.array(dataset.reviews_pre)
        x_post = np.array(dataset.reviews_post)
        y = np.array(dataset.labels)
        
        # epoch마다 학습을 수행합니다.
        nsml_callback = Nsml_Callback()
        checkpoint = ModelCheckpoint('./best.hdf5', monitor='val_loss', save_best_only=True, mode='min', period=1)
#         dataset_val = MovieReviewDataset_val(DATASET_PATH, config.strmaxlen)
#         x_val = np.array(dataset_val.reviews)
#         y_val = np.array(dataset_val.labels)
        print("model training...")
        hist = model.fit(x_pre, [y,y,y], 
                         validation_split = 0.12,
                         batch_size=config.batch_size, callbacks=[nsml_callback, ], epochs=config.epochs, verbose=2)


    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'test/test_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)