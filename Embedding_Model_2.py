import pandas as pd
import os
import numpy as np

# test keras model
# keras libs
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, Embedding, Reshape
from keras.layers import concatenate

data_path = 'data'

def process(frame):
    Input_Store = np.array(frame.Store)
    Input_Dow = np.array(frame.DayOfWeek)

    Input_Year = np.array(frame.Date.dt.year)
    Input_Month = np.array(frame.Date.dt.month)
    Input_Day = np.array(frame.Date.dt.day)

    sales = np.array(frame.Sales)
    result = [Input_Store, Input_Dow, Input_Year, Input_Month, Input_Day]
    return(result, sales)

train_set = pd.read_csv(os.path.join(data_path, 'train.csv'),
                       parse_dates = ['Date'])

# TODO break this into train and validation sets for looking at the results
train_partition = train_set[train_set.Date < '2015-01-01']
validation_partition = train_set[train_set.Date >= '2015-01-01']

# define sets
Input_set, Input_Y = process(train_partition)
Validation_set, Validation_Y = process(validation_partition)


# keras model
input_vec = []
# store in
store_in = Input(shape=(1,))
embedding_1 = Embedding(1115, 50, input_length = 1)(store_in)
res_store = Reshape(target_shape=(50,))(embedding_1)
input_vec.append(store_in)

# day of week in
dow_in = Input(shape=(1,))
embedding_2 = Embedding(7, 6, input_length = 1)(dow_in)
res_dow = Reshape(target_shape=(6,))(embedding_2)
input_vec.append(dow_in)

# year
year_in = Input(shape = (1,))
embedding_yr = Embedding(3, 2, input_length = 1)(year_in)
res_yr = Reshape(target_shape=(2,))(embedding_yr)
input_vec.append(year_in)

# month
month_in = Input(shape = (1,))
embedding_month = Embedding(12, 6, input_length = 1)(month_in)
res_month = Reshape(target_shape=(6,))(embedding_month)
input_vec.append(month_in)

# day
day_in = Input(shape = (1,))
embedding_day = Embedding(31, 10, input_length = 1)(day_in)
res_day = Reshape(target_shape=(10,))(embedding_day)
input_vec.append(day_in)


# main
Merger = concatenate([res_store, res_dow, res_yr, res_month, res_day])
drop_1 = Dropout(0.2)(Merger)
dense_1 = Dense(100)(drop_1)
drop_2 = Dropout(0.2)(dense_1)

dense_out = Dense(1)(drop_2)

model = Model(inputs = input_vec, outputs = dense_out)
model.compile(loss='mae', optimizer='adam')

result = model.fit(Input_set, Input_Y, epochs = 100, batch_size = 128,
			validation_data=(Validation_set, Validation_Y), verbose=2, shuffle=False)

model.save('model/embedding_2.h5')
