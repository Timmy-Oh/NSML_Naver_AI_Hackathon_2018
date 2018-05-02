# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np

import nsml
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

### LGBM
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import gc
import lightgbm as lgb


### Custom
from dataset import MovieReviewDataset, word_preprocessor, char_preprocessor
from dataset import regexp, vect_fit, vect_transform, trn_val_seperation
##################################################################################   
##################################################################################   
##################################################################################   


##################################################################################   
##################################################################################   
##################################################################################   

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(models, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
#         model.save_weights(filename)
        # save model to file
        models[1].save_model('lgbm_model.txt')
        
        # dump model with pickle
        with open('lgbm.pkl', 'wb') as fout:
            pickle.dump(models[1], fout)
    

# # can predict with any iteration when loaded in pickle way
# y_pred = pkl_bst.predict(X_test, num_iteration=7)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
#         model.load_weights(filename)

        # load model with pickle to predict
        print('Model loading...')
        with open('model.pkl', 'rb') as fin:
            models[1] = pickle.load(fin)
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = vect_transform(raw_data, vect_word, vect_char)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        point = model.predict(preprocessed_data)
        point[point>10.] = 10.
        point[point<1.] = 1.

        point = point.squeeze(axis=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

# class Nsml_Callback(Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs, step=epoch)
#         nsml.save(epoch)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--strmaxlen', type=int, default=150)
    
#     args.add_argument('--maxlen', type=int, default=200)
    args.add_argument('--cell_size_l1', type=int, default=50)
    args.add_argument('--cell_size_l2', type=int, default=30)
    args.add_argument('--filter_size', type=int, default=32)
    args.add_argument('--kernel_size', type=int, default=2)
    args.add_argument('--embed_size', type=int, default=300)
    args.add_argument('--prob_dropout', type=float, default=0.5)
    args.add_argument('--max_features', type=int, default=251)
    args.add_argument('--batch_size', type=int, default=256)
    
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'
    
    print("model creating...")
    vect_word = TfidfVectorizer(ngram_range=(1,2), max_features=100000, preprocessor=word_preprocessor)
    vect_char = TfidfVectorizer(ngram_range=(2,4), max_features=100000, analyzer='char', preprocessor=char_preprocessor)
    
    model = object
    lgb_model = object
    
    models = (model, lgb_model, vect_word, vect_char)
    # DONOTCHANGE: Reserved for nsml use
    print("nsml binding...")
    bind_model(models, config)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        print("data loading...")
        dataset = MovieReviewDataset(DATASET_PATH)
#         X_trn, X_val, Y_trn, Y_val= trn_val_seperation(dataset, 144570)
        X_trn, X_val, Y_trn, Y_val= trn_val_seperation(dataset, 3)
        
        # Vectorizer를 학습합니다
        vect_word, vect_char = vect_fit(X_trn, vect_word, vect_char)
        
        # Text를 Vector화 합니다
        X_trn = vect_transform(X_trn, vect_word, vect_char)
        X_val = vect_transform(X_val, vect_word, vect_char)
        
        #Dataset 구성
        train_data = lgb.Dataset(X_trn, Y_trn)
        valid_data = lgb.Dataset(X_val, Y_val, reference=train_data)
        gc.collect()
        
        # params 세탕 합니다
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2'},
            'num_leaves': 7,
            'max_depth': 15,
            'learning_rate': 1.0,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1,
        }
        # 학습을 수행합니다.
#         dataset_val = MovieReviewDataset_val(DATASET_PATH, config.strmaxlen)
#         x_val = np.array(dataset_val.reviews)
#         y_val = np.array(dataset_val.labels)
        print("model training...")
        num_round= 50
        model = lgb.train(params, train_data, num_round, valid_sets=[valid_data])
        epoch = 0
        nsml.save(epoch)

    ###


    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)