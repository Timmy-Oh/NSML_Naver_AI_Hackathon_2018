# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np

# from sklearn.model_selection import KFold

# keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, add, multiply
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, Conv1D, Dropout
from keras.layers import Dropout, SpatialDropout1D, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, PReLU
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling1D
from keras.layers import K, Activation
from keras.engine import Layer

from keras.callbacks import Callback

import nsml
from dataset import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
import pickle as pkl


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, dataset, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        model.save_weights(filename)

        with open('dataset.pkl', 'wb') as f:
            pkl.dump(dataset, f)
            print("dataset is saved on nsml")
    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        model.load_weights(filename)
        print('Model loaded')
        with open('dataset.pkl', 'rb') as f:
            dataset = pkl.load(f)
        print("dataset is loaded")
        for i, k in enumerate(dataset.tokenizer.word_index.keys()):
            if i > 5000:
                print(k)
            if i == 5050:
                break

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        point = model.predict(preprocessed_data).squeeze(axis=1).tolist()
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
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--strmaxlen', type=int, default=30)
    
    args.add_argument('--maxlen', type=int, default=200)
    args.add_argument('--cell_size', type=int, default=64)
    args.add_argument('--cell_size2', type=int, default=32)
    args.add_argument('--embed_size', type=int, default=300)
    args.add_argument('--prob_dropout', type=float, default=0.4)
    args.add_argument('--max_features', type=int, default=251)
    args.add_argument('--batch_size', type=int, default=128)
    
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../data/movie_review/'
        
    def get_model(config):
        inp = Input(shape=(config.strmaxlen, ), name='input')
        emb = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
        x1 = SpatialDropout1D(config.prob_dropout)(emb)
        x1 = Bidirectional(CuDNNLSTM(config.cell_size, return_sequences=True))(x1)
#         x12 = Bidirectional(CuDNNGRU(config.cell_size, return_sequences=False))(x1)
        
#         x2 = SpatialDropout1D(config.prob_dropout)(emb)
#         x2 = Bidirectional(CuDNNGRU(config.cell_size2, return_sequences=True))(x2)
#         x22 = Bidirectional(CuDNNLSTM(config.cell_size2, return_sequences=False))(x1)
        
#         avg_pool1 = GlobalAveragePooling1D()(x1)
        max_pool1 = GlobalMaxPooling1D()(x1)
        
#         avg_pool12 = GlobalAveragePooling1D()(x12)
#         max_pool12 = GlobalMaxPooling1D()(x12)
        
#         avg_pool3 = GlobalAveragePooling1D()(x2)
#         max_pool3 = GlobalMaxPooling1D()(x2)
        
#         avg_pool14 = GlobalAveragePooling1D()(x22)
#         max_pool14 = GlobalMaxPooling1D()(x22)
        
#         conc = concatenate([avg_pool1, max_pool1, x12, avg_pool3, max_pool3, x22])
#         fc1 = Dense(50, activation='relu')(conc)
        fc1 = Dropout(config.prob_dropout)(max_pool1)
        outp = Dense(1)(fc1)
        
        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])
        
        return model
    
    print("model creating...")
    model = get_model(config)
    print("data loading...")
    dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
    print('of_train')
    for i, k in enumerate(dataset.tokenizer.word_index.keys()):
        if i > 5000:
            print(k)
        if i == 5050:
            break

    # DONOTCHANGE: Reserved for nsml use
    print("nsml binding...")
    bind_model(model=model, dataset=dataset, config=config)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        x = np.array(dataset.reviews)
        y = np.array(dataset.labels)
        
#         print(x[0:10])
        # epoch마다 학습을 수행합니다.
        nsml_callback = Nsml_Callback()
        print("model training...")
        hist = model.fit(x, y, 
                             batch_size=config.batch_size, callbacks=[nsml_callback], epochs=config.epochs, verbose=2)
    
#         kf = KFold(5, shuffle=True, random_state=1991)
#         for train_idx, valid_idx in kf.split(x, y):
#             hist = model.fit(x[train_idx], y[train_idx], validation_data=(x[valid_idx], y[valid_idx]),
#                              batch_size=config.batch_size, callbacks=[nsml_callback], epochs=config.epochs, verbose=2)
        
    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
#         print(res)