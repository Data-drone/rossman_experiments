
# load
import pickle
X_train = pickle.load(open( 'feat_table/xgb_1_x_train_table.pkl', "rb" ))
y_train = pickle.load(open( 'feat_table/xgb_1_y_train_table.pkl', "rb" ))
X_valid = pickle.load(open( 'feat_table/xgb_1_x_valid_table.pkl', "rb" ))
y_valid = pickle.load(open( 'feat_table/xgb_1_y_valid_table.pkl', "rb" ))
test = pickle.load(open( 'feat_table/xgb_1_test_table.pkl', "rb" ))
features = pickle.load(open( 'feat_table/xgb_1_features_vector_vector.pkl', "rb" ))

# keras libs
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout
import numpy as np

# reformatting
x_train_array = np.copy(X_train[features].values)
x_valid_array = np.copy(X_valid[features].values)

format_sub_train_x = x_train_array.reshape((x_train_array.shape[0], 1, x_train_array.shape[1]) )
format_sub_val_x = x_valid_array.reshape((x_valid_array.shape[0], 1, x_valid_array.shape[1]) )

# custom loss function
from keras import backend as K
def mean_squared_percentage_error(y_true, y_pred):
    return K.mean(K.square((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None)), axis=-1)
# mspe = MSPE = mean_squared_percentage_error

model_input = Input( shape = (format_sub_train_x.shape[1], format_sub_train_x.shape[2]) )
lstm_1 = LSTM(100, return_sequences=True)(model_input)
dropout_mod = Dropout(0.2)(lstm_1)
lstm_2 = LSTM(100)(dropout_mod)
dropout_mod_2 = Dropout(0.2)(lstm_2)
output = Dense(1)(dropout_mod_2)
model = Model(inputs = model_input, outputs = output)

# model convert to functional API
model.compile(loss=mean_squared_percentage_error, optimizer='adam')
model.save('model/LSTM_Model_6.h5')

# model 5 training
mod_5 = model.fit(format_sub_train_x, np.copy(y_train), epochs=350, batch_size=100, 
                    validation_data=(format_sub_val_x, np.copy(y_valid)), verbose=2, shuffle=False)

dtest = np.copy(test[features].values)
test_frame = dtest.reshape((dtest.shape[0], 1, dtest.shape[1]) )

test_probs = model.predict(test_frame)
# Make Submission

import pandas as pd
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs.flatten())})

result.to_csv("submission/lstm_model_6_submission.csv", index=False)

