# -*- coding: utf-8 -*-
def get_model_2rnn(
                  embedding_matrix, cell_size = 80, cell_type_GRU = True,
                  maxlen = 180, max_features = 100000, embed_size = 300,
                  prob_dropout = 0.2, emb_train = False
                 ):
    
    inp = Input(shape=(maxlen, ), name='input')

    ##pre
    x1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp)
    x1 = SpatialDropout1D(prob_dropout)(x1)
    
    if cell_type_GRU:
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
        x1 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1)
    else :
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
    
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
 
    ##merge
    conc = concatenate([avg_pool1, max_pool1])
    outp = Dense(1)(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])

    return model