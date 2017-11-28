import pandas as pd
import os
import numpy as np

# test keras model
# keras libs
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, Embedding, Reshape
from keras.layers import concatenate

data_path = 'data'

train_set = pd.read_csv(os.path.join(data_path, 'train.csv'))

Input_Store = np.array(train_set.Store)
Input_Dow = np.array(train_set.DayOfWeek)


Input_Y = np.array(train_set.Sales)

# keras model

# store in
store_in = Input(shape=(1,))
embedding_1 = Embedding(1115, 50, input_length = 1)(store_in)
res_store = Reshape(target_shape=(50,))(embedding_1)

# day of week in
dow_in = Input(shape=(1,))
embedding_2 = Embedding(7, 6, input_length = 1)(dow_in)
res_dow = Reshape(target_shape=(6,))(embedding_2)

# main
Merger = concatenate([res_store, res_dow])
drop_1 = Dropout(0.2)(Merger)
dense_1 = Dense(64)(drop_1)
drop_2 = Dropout(0.2)(dense_1)

dense_out = Dense(1)(drop_2)

model = Model(inputs = [store_in, dow_in], outputs = dense_out)
model.compile(loss='mae', optimizer='adam')

model.fit([Input_Store, Input_Dow], Input_Y, epochs = 10, batch_size = 10)
