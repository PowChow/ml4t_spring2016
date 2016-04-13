"""
1) Preprocess File for MC3 P2
KNN Stock Return Predictor and Trading Strategy

"""
import numpy as np
import math
import time
import datetime as dt
import pandas as pd

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import LinRegLearner as lrl
import BagLearner as bl
import KNNLearner as knn
from util import get_data, plot_data


def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    # Add 2sd above and below the rolling mean
    upper_band = rm + (2*rstd)
    lower_band = rm - (2*rstd)
    return upper_band, lower_band

def preprocess_data(sym, sdate, edate):

    df = get_data(sym, pd.date_range(sdate, edate)) #returns symbol with closing prices
    df.drop(['SPY'], axis=1, inplace=True)
    df['t-N'] = df[sym].shift(5)   # shifts days forward, look backwards 5 days
    df['t+N'] = df[sym].shift(-5)  # shifts days backwards, look forwards 5 days
    df['t+1'] = df[sym].shift(-1)  # shifts forwards, looks backwards 1 day

    # calculate 3 technical features -
    df['momentum'] = (df[sym]/df['t-N']) - 1.0            # measures the rise and fall of stock, strength and weakness of current price
    df['sma'] = pd.rolling_mean(df[sym], window=5)          # define current direction with a lag - smooth price action and filter out noise
    df['daily_ret'] = (df['t+N']/df[sym]) - 1.0           # change in price in 5 days
    df['vol'] = pd.rolling_std(df['daily_ret'], window=5)   # amont of variability or dispersion around the average -- evaluate risk and signifigance of price movement
    df['bb_upper_band'] = df['sma'] +(2*df['vol'])
    df['bb_lower_band'] = df['sma'] - (2*df['vol'])
    df['bb'] = (df[sym] - df['sma'])/((df['bb_upper_band']-df['bb_lower_band'])/2)  #range of volatility


    df['momentum_norm'] = (df['momentum'] - df['momentum'].min())/ (df['momentum'].max() - df['momentum'].min())
    df['sma_norm'] = (df['sma'] - df['sma'].min())/ (df['sma'].max() - df['sma'].min())
    df['vol_norm'] = (df['vol'] - df['vol'].min())/ (df['vol'].max() - df['vol'].min())
    df['bb_norm'] =  (df['bb'] - df['bb'].min())/ (df['bb'].max() - df['bb'].min())

    df.dropna(how='any', inplace=True)

    #printing for debugging
    print df.head(10)
    print df.describe()
    #print data.dtypes

    return df



if __name__ == "__main__":
    # 1) Set constants
    allocations = [0.2, 0.2, 0.4, 0.2] # example from previous files
    start_val = 1000000  # default from previous assignments
    risk_free_rate = 0.0 # default from previous assignments
    sample_freq = 252    # default from previous assignments

    # 2) Get and process training set
    #sym = ['IBM']
    sym = ['ML4T-220']

    train_data = preprocess_data(sym=sym, sdate=dt.datetime(2007, 12, 31),
                                 edate=dt.datetime(2009, 12, 31))
    train = train_data[['momentum_norm', 'sma_norm', 'vol_norm', 'bb_norm', 'daily_ret']]
    train.to_csv('Data/%s_train_norms.csv' % sym[0], index=False,encoding='utf-8', header=False)

    test_data = preprocess_data(sym=sym, sdate=dt.datetime(2009, 12, 31),
                                 edate=dt.datetime(2011, 12, 31))
    test = test_data[['momentum_norm', 'sma_norm', 'vol_norm', 'bb_norm', 'daily_ret']]
    test.to_csv('Data/%s_test_norms.csv' % sym[0],index=False,encoding='utf-8', header=False)

    # 3) Send test and train data to model
    trainX = np.array(train.iloc[1:,0:-1])
    trainY = np.array(train.iloc[1:,-1])
    testX = np.array(test.iloc[1:,0:-1])
    testY = np.array(test.iloc[1:,-1])

    # print trainX.shape
    # print trainY.shape
    # print testX.shape
    # print testY.shape
    # print testY[0]


    model = 'knn'
    learner = knn.KNNLearner(k=3, verbose=True) # create a knnLearner
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in sample
    predY_train = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY_train) ** 2).sum()/trainY.shape[0])
    print model
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY_train, y=trainY)
    print "corr: ", c[0,1]
    #create graph in sample comparing predicted and actual
    #plot_scatter(predY_train, trainY, 'in_sample', model)

    # evaluate out of sample
    predY_test = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY_test) ** 2).sum()/testY.shape[0])
    print
    print model
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY_test, y=testY)
    print "corr: ", c[0,1]

    print predY_test
    print len(predY_test)
    # 4) generate orders from KNNLearner predicted values

    #output first graphs



#market sim
# create function to create csv of orders for back testing
# send to trading simulator / back testing
# calcuate 5-day relative return, Y


# send testing set to test predictions

#output graph


