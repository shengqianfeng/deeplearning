#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : lstm_stock_price_train_of_keras.py
@Author : jeffsheng
@Date : 2020/1/17
@Desc : 使用keras预测股票价格模型
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math, time
import sklearn
import sklearn.preprocessing
from pandas import datetime
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
import h5py
import requests
import os

test_set_size_percentage = 10

# display parent directory and working directory
print(os.path.dirname(os.getcwd())+':', os.listdir(os.path.dirname(os.getcwd())));
print(os.getcwd()+':', os.listdir(os.getcwd()));

df = pd.read_csv("./temp/prices-split-adjusted.csv", index_col = 0)

# number of different stocks:  501
print('\nnumber of different stocks: ', len(list(set(df.symbol))))
# ['EVHC', 'NEM', 'FB', 'OMC', 'LLTC', 'COTY', 'HRB', 'EFX', 'LB', 'XOM']
print(list(set(df.symbol))[:10])
print(df.head())

print(df.shape)

print(df.isnull().sum())


# 定义min-max标准化的函数
def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))
    return df


# 直接将数据划分为训练集和测试集
def load_data_keras(stock, seq_len):
    data_raw = stock.as_matrix()
    data = []

    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])

    data = np.array(data);

    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - test_set_size;

    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]

    x_test = data[train_set_size:,:-1,:]
    y_test = data[train_set_size:,-1,:]

    return [x_train, y_train, x_test, y_test]


# 选择 EQIX 这一支
df_stock = df[df.symbol == 'EQIX'].copy()
print(df_stock.head())

print("--------------可视化EQIX价格走势----------------")
global closing_stock
global opening_stock

f, axs = plt.subplots(2,2,figsize=(8,8))
plt.subplot(212)

company = df[df['symbol']=='EQIX']
company = company.open.values.astype('float32')
company = company.reshape(-1, 1)
opening_stock = company

plt.grid(True)
plt.xlabel('Time')
plt.ylabel('EQIX' + " open stock prices")
plt.title('prices Vs Time')
plt.plot(company , 'g')

plt.subplot(211)
company_close = df[df['symbol']=='EQIX']
company_close = company_close.close.values.astype('float32')
company_close = company_close.reshape(-1, 1)
closing_stock = company_close

plt.xlabel('Time')
plt.ylabel('EQIX' + " close stock prices")
plt.title('prices Vs Time')
plt.grid(True)
plt.plot(company_close , 'b')
plt.show()

# 删掉 symbol 和 volume 两列，只保留四个价格
df_stock.drop(['symbol'],1,inplace=True)
df_stock.drop(['volume'],1,inplace=True)
cols = list(df_stock.columns.values)
print('df_stock.columns.values = ', cols)
print(df_stock.head())


print("-----------获取训练集和测试集-----------------------")
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)

seq_len = 20
x_train_keras, y_train_keras, x_test_keras, y_test_keras = load_data_keras(df_stock_norm, seq_len)

# 构建模型函数
def build_model(layers):
    """
    输入参数layers：[4, seq_len-1, 4]
    ① 第一个4是指x有4列
    ② seq_len-1是每一批即x_i的长度，每个x_i有19条数据
    ③ 第二个 4 是 y 有 4 列
    :param layers:
    :return:
    """
    d = 0.3
    model = Sequential()

    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(4,kernel_initializer="uniform",activation='linear'))

    start = time.time()
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model


model = build_model([4,seq_len-1,4])


print("-------------训练模型------------------")
model.fit(x_train_keras,y_train_keras,batch_size=50,epochs=100,validation_split=0.1,verbose=1)


def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]


model_score(model, x_train_keras, y_train_keras, x_test_keras, y_test_keras)


