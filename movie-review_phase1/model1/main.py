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
from dataset import MovieReviewDataset, preprocess
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
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        point = model.predict(preprocessed_data)[3]
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

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=35)
    args.add_argument('--strmaxlen', type=int, default=300)
    
    args.add_argument('--cell_size', type=int, default=64)
    args.add_argument('--cell_size2', type=int, default=50)
    args.add_argument('--embed_size', type=int, default=300)
    args.add_argument('--prob_dropout', type=float, default=0.4)
    args.add_argument('--max_features', type=int, default=251)
    args.add_argument('--batch_size', type=int, default=256)
    
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../data/movie_review/'
        
    def get_model(config):
        #model
        #wrote out all the blocks instead of looping for simplicity
        filter_nr = 64
        filter_size = 3
        max_pool_size = 3
        max_pool_strides = 2
        dense_nr = 64
        spatial_dropout = 0.3
        dense_dropout = 0.4
        conv_kern_reg = regularizers.l2(0.000005)
        conv_bias_reg = regularizers.l2(0.000005)
        

        inp = Input(shape=(config.strmaxlen, ), name='input')
        emb = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
        
        emb1 = SpatialDropout1D(config.prob_dropout)(emb)

        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb1)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)
        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)

        #we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
        #if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
        resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb1)
        resize_emb = PReLU()(resize_emb)

        block1_output = add([block1, resize_emb])
        block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

        block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
        block2 = BatchNormalization()(block2)
        block2 = PReLU()(block2)
        block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
        block2 = BatchNormalization()(block2)
        block2 = PReLU()(block2)

        block2_output = add([block2, block1_output])
        block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

        block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
        block3 = BatchNormalization()(block3)
        block3 = PReLU()(block3)
        block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
        block3 = BatchNormalization()(block3)
        block3 = PReLU()(block3)

        block3_output = add([block3, block2_output])
        block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

        block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
        block4 = BatchNormalization()(block4)
        block4 = PReLU()(block4)
        block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
        block4 = BatchNormalization()(block4)
        block4 = PReLU()(block4)

        block4_output = add([block4, block3_output])
        block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

        block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
        block5 = BatchNormalization()(block5)
        block5 = PReLU()(block5)
        block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
        block5 = BatchNormalization()(block5)
        block5 = PReLU()(block5)

        block5_output = add([block5, block4_output])
        block5_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

        block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
        block6 = BatchNormalization()(block6)
        block6 = PReLU()(block6)
        block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
        block6 = BatchNormalization()(block6)
        block6 = PReLU()(block6)

        block6_output = add([block6, block5_output])
        block6_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)

        block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6_output)
        block7 = BatchNormalization()(block7)
        block7 = PReLU()(block7)
        block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block7)
        block7 = BatchNormalization()(block7)
        block7 = PReLU()(block7)

        block7_output = add([block7, block6_output])
        output = GlobalMaxPooling1D()(block7_output)

        output = Dense(dense_nr, activation='linear')(output)
        output = BatchNormalization()(output)
        output = PReLU()(output)
        output = Dropout(dense_dropout)(output)
        dpcnn_out = Dense(1)(output)

#         model = Model(inputs=inp, outputs=output)
#         model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])
    
        
###         ========================================================
#         inp = Input(shape=(config.strmaxlen, ), name='input')
#         emb = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
        emb2 = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
        x1 = SpatialDropout1D(config.prob_dropout)(emb2)
        x1 = Bidirectional(CuDNNLSTM(config.cell_size, return_sequences=True))(x1)
        x12 = Bidirectional(CuDNNGRU(config.cell_size, return_sequences=True))(x1)
        x12c = Conv1D(filter_nr, kernel_size = filter_size, strides=1, padding = "valid", kernel_initializer = "he_uniform")(x12)
#         x2 = SpatialDropout1D(config.prob_dropout)(emb)
#         x2 = Bidirectional(CuDNNGRU(config.cell_size2, return_sequences=True))(x2)
#         x22 = Bidirectional(CuDNNLSTM(config.cell_size2, return_sequences=False))(x1)
        
        avg_pool1 = GlobalAveragePooling1D()(x1)
        max_pool1 = GlobalMaxPooling1D()(x1)
        
        avg_pool12 = GlobalAveragePooling1D()(x12)
        max_pool12 = GlobalMaxPooling1D()(x12)
        
        avg_pool12c = GlobalAveragePooling1D()(x12c)
        max_pool12c = GlobalMaxPooling1D()(x12c)
        
#         avg_pool14 = GlobalAveragePooling1D()(x22)
#         max_pool14 = GlobalMaxPooling1D()(x22)
        
        conc = concatenate([avg_pool1, max_pool1, avg_pool12, max_pool12, avg_pool12c, max_pool12c])
#         fc1 = Dense(50, activation='relu')(conc)
        fc1 = Dropout(config.prob_dropout)(conc)
        rnnc_out = Dense(1)(fc1)
        
#         model = Model(inputs=inp, outputs=outp)
#         model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])
        
# #         ==================================================================================================

###         ========================================================
#         inp = Input(shape=(config.strmaxlen, ), name='input')
        emb3 = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
        r1 = SpatialDropout1D(config.prob_dropout)(emb3)
        r1 = Bidirectional(CuDNNLSTM(config.cell_size2, return_sequences=True))(r1)
        r12 = Bidirectional(CuDNNLSTM(config.cell_size2, return_sequences=False))(r1)
      
        rfc1 = Dense(50, activation='relu')(r12)
        rfc1 = Dropout(config.prob_dropout)(rfc1)
        rnn_out = Dense(1)(rfc1)
        
#         model = Model(inputs=inp, outputs=outp)
#         model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])
        
# #         ==================================================================================================
        ens_out = average([rnn_out, rnnc_out, dpcnn_out])
        model = Model(inputs=inp, outputs=[rnn_out, rnnc_out, dpcnn_out, ens_out])
        model.compile(loss='mean_squared_error', optimizer='adam', loss_weights=[1.,0.8,1.,0.3], metrics=['mean_squared_error', 'accuracy'])
        
        return model
    
    print("model creating...")
    model = get_model(config)
    model.summary()
    
#     print('of_train')
#     for i, k in enumerate(dataset.tokenizer.word_index.keys()):
#         if i > 5000:
#             print(k)
#         if i == 5050:
#             break

    # DONOTCHANGE: Reserved for nsml use
    print("nsml binding...")
    bind_model(model=model, config=config)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        print("data loading...")
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        x = np.array(dataset.reviews)
        y = np.array(dataset.labels)
        
#         print(x[0:10])
        # epoch마다 학습을 수행합니다.
    
        def schedule(ind):
            a = [0.005, 0.005, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0005, 0.0005,
                 0.004, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.0005,
                 0.003, 0.002, 0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0001, 0.0001]
            return a[ind]
        lr_s = LearningRateScheduler(schedule)
        
        nsml_callback = Nsml_Callback()
        print("model training...")
        hist = model.fit(x, [y,y,y,y], 
                         validation_split=0.12,
                             batch_size=config.batch_size, callbacks=[nsml_callback], epochs=config.epochs, verbose=2)
    
#         kf = KFold(5, shuffle=True, random_state=1991)
#         for train_idx, valid_idx in kf.split(x, y):
#             hist = model.fit(x[train_idx], y[train_idx], validation_data=(x[valid_idx], y[valid_idx]),
#                              batch_size=config.batch_size, callbacks=[nsml_callback], epochs=config.epochs, verbose=2)
        
    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join('../sample_data/movie_review/', 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)