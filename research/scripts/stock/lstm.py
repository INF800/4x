# ==============================================================================================
# beg: basic imports and setup
# ==============================================================================================
from datetime import datetime
from loguru import logger

import numpy as np
import pandas as pd
from pandas_datareader.data import DataReader
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
# ==============================================================================================
# end: basic imports and setup
# ==============================================================================================


# ==============================================================================================
# beg: config
# ==============================================================================================
STOCK_NAME = 'KIRK'
YEARS = 20
END_DATE = datetime.now()
BEG_DATE = datetime(END_DATE.year - YEARS, END_DATE.month, END_DATE.day)

WIN_SIZE = 64
TRAIN_TEST_RAT = 0.7

EPOCHS = 5
BATCH_SIZE = 512
# ==============================================================================================
# end: config
# ==============================================================================================


# ==============================================================================================
# beg: data preparation
# ==============================================================================================
scaler = MinMaxScaler(feature_range=(0,1))

def get_stock(stock, beg, end):
    """ return df from yfinance """
    return DataReader(stock, 'yahoo', beg, end)


def pre_proc(df):
    """ 
    generate trainable data 
    """
    
    def __create_roll_win(scaled_values, size=60):
        single_col_df = pd.DataFrame({'ScaledClose': scaled_values.flatten()})
        _df = pd.concat([single_col_df.shift(i) for i in range(size, -1, -1)], axis=1)
        return _df.dropna().values

    def __scale(single_col_df):
        return scaler.fit_transform(single_col_df)

    cls_vals = df.dropna().filter(['Close'])
    scl_vals = __scale(cls_vals)
    sld_wins = __create_roll_win(scl_vals, WIN_SIZE)

    xs = sld_wins[:, :WIN_SIZE]
    ys = sld_wins[:,  WIN_SIZE]

    return xs, ys


data = get_stock(stock=STOCK_NAME, beg=BEG_DATE, end=END_DATE)
xs, ys = pre_proc(df=data)

split_idx = int(TRAIN_TEST_RAT*len(xs))
test_size = int((1-TRAIN_TEST_RAT)*len(xs))

x_train, y_train = xs[:split_idx], ys[:split_idx]
x_valid, y_valid = xs[split_idx: split_idx+test_size//2], ys[split_idx: split_idx+test_size//2]
x_hdout, y_hdout = xs[split_idx+test_size//2:], ys[split_idx+test_size//2:]
#x_test, y_test = xs[split_idx:], ys[split_idx:]

# add feature axis
x_train = np.expand_dims(x_train, 2)
x_valid = np.expand_dims(x_valid, 2)
x_hdout = np.expand_dims(x_hdout, 2)
#x_test = np.expand_dims(x_test, 2)
# ==============================================================================================
# end: data preparation
# ==============================================================================================


# ==============================================================================================
# beg: modelling
# ==============================================================================================
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(
    x_train, y_train, 
    validation_data=(x_valid, y_valid),
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS,
)

# eval on unseen
predictions = model.predict(x_hdout)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(((predictions.flatten() - y_hdout.flatten()) ** 2)))
logger.info('='*80)
logger.info(f'[EVAL] RMSE = {rmse}')
logger.info('='*80)
# ==============================================================================================
# end: modelling
# ==============================================================================================


# ==============================================================================================
# beg: plot results
# ==============================================================================================
data = data.dropna().filter(['Close'])

train = data[WIN_SIZE:WIN_SIZE+len(y_train)]
valid = data[WIN_SIZE+len(y_train):WIN_SIZE+len(y_train)+len(y_valid)]
hdout = data[WIN_SIZE+len(y_train)+len(y_valid):WIN_SIZE+len(y_train)+len(y_valid)+len(x_hdout)]

logger.info(f'[SIZE] data: {len(data)}')
logger.info(f'[SIZE] train: {len(train)} \tx_train: {x_train.shape}')
logger.info(f'[SIZE] valid: {len(valid)} \tx_valid: {x_valid.shape}')
logger.info(f'[SIZE] hdout: {len(hdout)} \tx_hdout: {x_hdout.shape}')
logger.info(f'[SIZE] train+valid+hdout-WIN_SIZE {len(hdout) + len(valid) + len(train) - WIN_SIZE}')

predictions = model.predict(x_valid)
predictions = scaler.inverse_transform(predictions)
valid['Predictions'] = predictions

predictions = model.predict(x_hdout)
predictions = scaler.inverse_transform(predictions)
hdout['Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.plot(hdout[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Val Preds', 'Holdout', 'Holdout Preds'], loc='lower right')
plt.show()
# ==============================================================================================
# end: plot results
# ==============================================================================================