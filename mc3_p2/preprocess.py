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
import matplotlib.pyplot as plt
from util import get_data, plot_data


def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return pd.rolling_mean(values, window=window)

def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return pd.rolling_std(values, window=window)

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
    sym = ['IBM'] #use this one for graded plot
    dates = pd.date_range(start_date, end_date)
    data = get_data(sym, dates) #returns symbol with closing prices
    SPY = data['SPY']
    data.drop(['SPY'], axis=1, inplace=True)
    data_shift = data.shift(-5)

    # calculate 3 technical features -
    #data['bb'] =
    data['momentum'] = (data[sym]/data_shift[sym]) - 1.0
    #data['risk'] =
    data['Y'] = (data_shift[sym]/data[sym]) - 1.0

    data['sma'] = get_rolling_mean(data[sym], window=5)

    # 2. Compute rolling standard deviation
    data['std'] = get_rolling_std(data[sym], window=5)

    #get bollinger band averages
    #data['bb'] = (data[sym] - data['sma'])/(2 * data['std'])
    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(data['sma'], data['std'])


    print data.head(10)
    print data.columns
# normalize these tech values
# only output every 5th technical value, rows[::5]

#out put technical features to file with normalized tech values

# calcuate 5-day relative return, Y

# send training set with bagging to test predictions

# output graphs

# send testing set to test predictions

#output graph

# send to trading simulator / back testing
