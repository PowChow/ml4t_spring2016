"""
1) Preprocess File for MC3 P2
KNN Stock Return Predictor and Trading Strategy

"""
import numpy as np
import math
import time

import LinRegLearner as lrl
import BagLearner as bl
import KNNLearner as knn
import matplotlib.pyplot as plt
from util import get_data, plot_data


if __name == "__main__":
    # Set constants
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2010, 12, 31)
    allocations = [0.2, 0.2, 0.4, 0.2] # use this for grading
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    #Read Files
    symbols = ['IBM'] #use this one for graded plot
    prices_all = get_data(syms, dates)

# calcluate 3 technical features

#out put technical features to file

# calcuate 5-day relative return, Y

# send training set with bagging to test predictions

# output graphs

# send testing set to test predictions

#output graph

# send to trading simulator / back testing
