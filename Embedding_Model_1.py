import pandas as pd
import os
import numpy as np

# test keras model
# keras libs
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, Embedding, Reshape

data_path = 'data'

train_set = pd.read_csv(os.path.join(data_path, 'train.csv'))

Input_X = np.array(train_set.Store)
Input_Y = np.array(train_set.Sales)

# keras model
model_in = Input(shape=(1,))
embedding_1 = Embedding(1115, 50, input_length = 1)(model_in)
res = Reshape(target_shape=(50,))(embedding_1)
dense_1 = Dense(1)(res)
model = Model(inputs = model_in, outputs = dense_1)
model.compile(loss='mae', optimizer='adam')

model.fit(Input_X, Input_Y, epochs = 10, batch_size = 10)
