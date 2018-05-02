
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import tensorflow as tf
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess, get_tknzr
from sklearn.model_selection import train_test_split
# keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, Conv1D
from keras.layers import Dropout, SpatialDropout1D, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam, RMSprop, Nadam
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, add, multiply, average
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, Conv1D, Dropout
from keras.layers import Dropout, SpatialDropout1D, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, PReLU
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling1D
from keras.layers import K, Activation
from keras.engine import Layer
from keras.models import Model
from keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, SpatialDropout1D
from keras.layers import add, Dropout, PReLU, BatchNormalization, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import optimizers
from keras import initializers, regularizers, constraints, callbacks
import pickle as pkl
def bind_model(model, config, tknzr):
    def save(filename, *args):
        model.save_weights(filename)
        with open(os.path.join(filename, 'tknzr.pkl'), 'wb') as f:
            pkl.dump(tknzr, f)
    def load(filename, *args):
        model.load_weights(filename)
        with open(os.path.join(filename, 'tknzr.pkl'), 'rb') as f:
            tknzr = pkl.load(f)

    def infer(raw_data, **kwargs):
        
        preprocessed_data = preprocess(raw_data, tknzr, config.strmaxlen)
        pred = model.predict(preprocessed_data)[-1]
        pred_prob = pred[:,1]
        clipped = np.argmax(pred, axis=-1)
        return list(zip(pred_prob.flatten(), clipped.flatten()))
    nsml.bind(save=save, load=load, infer=infer)

class Nsml_Callback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs, step=epoch)
        nsml.save(epoch)
        
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
    args.add_argument('--epochs', type=int, default=20)
    args.add_argument('--strmaxlen', type=int, default=80)
    
    args.add_argument('--cell_size_l1', type=int, default=50)
    args.add_argument('--cell_size_l2', type=int, default=40)
    args.add_argument('--filter_size', type=int, default=32)
    args.add_argument('--kernel_size', type=int, default=3)
    args.add_argument('--embed_size', type=int, default=256)
    args.add_argument('--prob_dropout', type=float, default=0.4)
    args.add_argument('--prob_dropout2', type=float, default=0.2)
    args.add_argument('--max_features', type=int, default=11200)
    args.add_argument('--batch_size', type=int, default=20)
    
    args.add_argument('--verbose', type=int, default=2)
    args.add_argument('--random_state', type=int, default=55)
    args.add_argument('--test_size', type=float, default=0.05)
    args.add_argument('--validation', type=bool, default=False)
    
    config = args.parse_args()
    
    
    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = './sample_data/kin'

    def get_model(config):
        inp = Input(shape=(config.strmaxlen, ), name='input')

        emb = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
        emb1 = SpatialDropout1D(config.prob_dropout)(emb)
        #### 
        l1_L = Bidirectional(CuDNNLSTM(config.cell_size_l1, return_sequences=True))(emb1)
        l2_LL = Bidirectional(CuDNNLSTM(config.cell_size_l2, return_sequences=True))(l1_L)
        l2_LG = Bidirectional(CuDNNGRU(config.cell_size_l2, return_sequences=True))(l1_L)
        l3_LLC = Conv1D(config.filter_size, kernel_size = config.kernel_size, strides=2, padding = "valid", kernel_initializer = "he_uniform")(l2_LL)
        l3_LGC = Conv1D(config.filter_size, kernel_size = config.kernel_size, strides=2, padding = "valid", kernel_initializer = "he_uniform")(l2_LG)

        avg_pool_L = GlobalAveragePooling1D()(l1_L)
        max_pool_L = GlobalMaxPooling1D()(l1_L)
        avg_pool_LL = GlobalAveragePooling1D()(l2_LL)
        max_pool_LL = GlobalMaxPooling1D()(l2_LL)
        avg_pool_LG = GlobalAveragePooling1D()(l2_LG)
        max_pool_LG = GlobalMaxPooling1D()(l2_LG)
        attention_LLA = Attention(config.strmaxlen)(l2_LL)
        attention_LGA = Attention(config.strmaxlen)(l2_LG)
        avg_pool_LLC = GlobalAveragePooling1D()(l3_LLC)
        max_pool_LLC = GlobalMaxPooling1D()(l3_LLC)
        avg_pool_LGC = GlobalAveragePooling1D()(l3_LGC)
        max_pool_LGC = GlobalMaxPooling1D()(l3_LGC)
        attention_LLCA = Attention(int(config.strmaxlen/2-1))(l3_LLC)
        attention_LGCA = Attention(int(config.strmaxlen/2-1))(l3_LGC)
        conc_LLC = concatenate([avg_pool_L, max_pool_L, avg_pool_LL, max_pool_LL, avg_pool_LLC, max_pool_LLC, attention_LLA, attention_LLCA])
        conc_LGC = concatenate([avg_pool_L, max_pool_L, avg_pool_LG, max_pool_LG, avg_pool_LGC, max_pool_LGC, attention_LGA, attention_LGCA])        
        out_LL = Dropout(config.prob_dropout2)(conc_LLC)
        out_LG = Dropout(config.prob_dropout2)(conc_LGC)
        out_LL = Dense(2, activation='softmax')(out_LL)
        out_LG = Dense(2, activation='softmax')(out_LG)
        ####
        
#         emb2 = Embedding(config.max_features, config.max_features,embeddings_initializer='identity', trainable = True)(inp)
#         emb1 = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
        emb2 = SpatialDropout1D(config.prob_dropout)(emb)
        
        #### 
        l1_G = Bidirectional(CuDNNGRU(config.cell_size_l1, return_sequences=True))(emb2)
        
        l2_GL = Bidirectional(CuDNNLSTM(config.cell_size_l2, return_sequences=True))(l1_G)
        l2_GG = Bidirectional(CuDNNGRU(config.cell_size_l2, return_sequences=True))(l1_G)
        
        l3_GLC = Conv1D(config.filter_size, kernel_size = config.kernel_size, strides=2, padding = "valid", kernel_initializer = "he_uniform")(l2_GL)
        l3_GGC = Conv1D(config.filter_size, kernel_size = config.kernel_size, strides=2, padding = "valid", kernel_initializer = "he_uniform")(l2_GG)

        avg_pool_G = GlobalAveragePooling1D()(l1_G)
        max_pool_G = GlobalMaxPooling1D()(l1_G)
        
        
        avg_pool_GL = GlobalAveragePooling1D()(l2_GL)
        max_pool_GL = GlobalMaxPooling1D()(l2_GL)
        avg_pool_GG = GlobalAveragePooling1D()(l2_GG)
        max_pool_GG = GlobalMaxPooling1D()(l2_GG)
        
        attention_GLA = Attention(config.strmaxlen)(l2_GL)
        attention_GGA = Attention(config.strmaxlen)(l2_GG)

        avg_pool_GLC = GlobalAveragePooling1D()(l3_GLC)
        max_pool_GLC = GlobalMaxPooling1D()(l3_GLC)
        avg_pool_GGC = GlobalAveragePooling1D()(l3_GGC)
        max_pool_GGC = GlobalMaxPooling1D()(l3_GGC)
        
        attention_GLCA = Attention(int(config.strmaxlen/2-1))(l3_GLC)
        attention_GGCA = Attention(int(config.strmaxlen/2-1))(l3_GGC)
        
        conc_GLC = concatenate([avg_pool_G, max_pool_G, avg_pool_GL, max_pool_GL, avg_pool_GLC, max_pool_GLC, attention_GLA, attention_GLCA])
        conc_GGC = concatenate([avg_pool_G, max_pool_G, avg_pool_GG, max_pool_GG, avg_pool_GGC, max_pool_GGC, attention_GGA, attention_GGCA])        

        out_GL = Dropout(config.prob_dropout2)(conc_GLC)
        out_GG = Dropout(config.prob_dropout2)(conc_GGC)
        out_GL = Dense(2, activation='softmax')(out_GL)
        out_GG = Dense(2, activation='softmax')(out_GG)
        
        out_avg = average([out_LL, out_LG, out_GL, out_GG])

        
# #         ==================================================================================================
        model = Model(inputs=inp, outputs=[out_LL, out_LG, out_GL, out_GG, out_avg])
        model.compile(loss='categorical_crossentropy', optimizer='adam', loss_weights=[1., 1., 1., 1., 0.1], metrics=['accuracy'])
        return model
    
    
    tknzr = object

    model = get_model(config)
    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config, tknzr)
    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        tknzr = get_tknzr(dataset.queries)
        q = preprocess(dataset.queries, tknzr, config.strmaxlen)

        print(config)
        nsml_callback = Nsml_Callback()
        x = np.array(q)
        y = np.array(dataset.labels)
        y = to_categorical(y, num_classes=2)
        if config.validation:
            hist = model.fit(x, [y,y,y,y,y], batch_size=config.batch_size, callbacks=[nsml_callback], epochs=config.epochs, verbose=2, validation_split=0.3)
        else:
            hist = model.fit(x, [y,y,y,y,y], batch_size=config.batch_size, callbacks=[nsml_callback], epochs=config.epochs, verbose=2)

    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/sample'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)