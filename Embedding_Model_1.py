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

Input_Year = np.array(train_set.Date.dt.year)
Input_Month = np.array(train_set.Date.dt.month)
Input_Day = np.array(train_set.Date.dt.day)

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

# year
year_in = Input(shape = (1,))
embedding_yr = Embedding(3, 2, input_length = 1)(year_in)
res_yr = Reshape(target_shape=(2,))(embedding_yr)

# month
month_in = Input(shape = (1,))
embedding_month = Embedding(12, 6, input_length = 1)(month_in)
res_month = Reshape(target_shape=(6,))(embedding_month)

# day
day_in = Input(shape = (1,))
embedding_day = Embedding(31, 10, input_length = 1)(day_in)
res_day = Reshape(target_shape=(10,))(embedding_day)

# main
Merger = concatenate([res_store, res_dow, res_yr, res_month, res_day])
drop_1 = Dropout(0.2)(Merger)
dense_1 = Dense(100)(drop_1)
drop_2 = Dropout(0.2)(dense_1)

dense_out = Dense(1)(drop_2)

model = Model(inputs = [store_in, dow_in, year_in, month_in, day_in], outputs = dense_out)
model.compile(loss='mae', optimizer='adam')

result = model.fit([Input_Store, Input_Dow, Input_Year, Input_Month, Input_Day], Input_Y, epochs = 15, batch_size = 128)

model.save('model/embedding_2.h5')
