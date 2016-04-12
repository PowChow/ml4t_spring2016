"""
1) Preprocess File for MC3 P2
KNN Stock Return Predictor and Trading Strategy

"""
import numpy as np
import math
import time
import datetime as dt
import pandas as pd

import LinRegLearner as lrl
import BagLearner as bl
import KNNLearner as knn
from util import get_data, plot_data

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    # Add 2sd above and below the rolling mean
    upper_band = rm + (2*rstd)
    lower_band = rm - (2*rstd)
    return upper_band, lower_band


if __name__ == "__main__":
    # Set constants
    start_date = dt.datetime(2007, 12, 31)
    end_date = dt.datetime(2009, 12, 31)
    allocations = [0.2, 0.2, 0.4, 0.2] # example from previous files
    start_val = 1000000  # default from previous assignments
    risk_free_rate = 0.0 # default from previous assignments
    sample_freq = 252    # default from previous assignments

    #Read Files
    #sym = ['IBM']
    sym = ['ML4T-220']
    dates = pd.date_range(start_date, end_date)
    data = get_data(sym, dates) #returns symbol with closing prices
    SPY = data['SPY']
    data.drop(['SPY'], axis=1, inplace=True)
    data['t-N'] = data[sym].shift(5)   # shifts days forward, look backwards 5 days
    data['t+N'] = data[sym].shift(-5)  # shifts days backwards, look forwards 5 days
    data['t+1'] = data[sym].shift(-1)  # shifts forwards, looks backwards 1 day

    # calculate 3 technical features -
    data['momentum'] = (data[sym]/data['t-N']) - 1.0            # measures the rise and fall of stock, strength and weakness of current price
    data['sma'] = pd.rolling_mean(data[sym], window=5)          # define current direction with a lag - smooth price action and filter out noise
    data['daily_ret'] = (data['t+N']/data[sym]) - 1.0           # change in price in 5 days
    data['vol'] = pd.rolling_std(data['daily_ret'], window=5)   # amont of variability or dispersion around the average -- evaluate risk and signifigance of price movement
    data['bb_upper_band'] = data['sma'] +(2*data['vol'])
    data['bb_lower_band'] = data['sma'] - (2*data['vol'])
    data['bb'] = (data[sym] - data['sma'])/((data['bb_upper_band']-data['bb_lower_band'])/2)  #range of volatility


    data['momentum_norm'] = (data['momentum'] - data['momentum'].min())/ (data['momentum'].max() - data['momentum'].min())
    data['sma_norm'] = (data['sma'] - data['sma'].min())/ (data['sma'].max() - data['sma'].min())
    data['vol_norm'] = (data['vol'] - data['vol'].min())/ (data['vol'].max() - data['vol'].min())
    data['bb_norm'] =  (data['bb'] - data['bb'].min())/ (data['bb'].max() - data['bb'].min())

    data.dropna(how='any', inplace=True)

    print data.head(10)
    print data.describe()
    #print data.dtypes

    data[['momentum_norm', 'sma_norm', 'vol_norm', 'bb_norm', 'daily_ret']].to_csv('Data/%s_norms.csv' % sym[0],
                                                                                   index=False,
                                                                                   encoding='utf-8',
                                                                                   heading=False
                                                                                   )
    # only output every 5th technical value, rows[::5], not sure if it this is necessary anymore

    # output technical features to file with normalized tech values

    #output first graphs


# send training set with bagging to test predictions

# output graphs

# create function to create csv of orders for back testing
# send to trading simulator / back testing
# calcuate 5-day relative return, Y


# send testing set to test predictions

#output graph


