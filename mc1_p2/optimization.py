"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
#function finds the optimal allocation for a given set of stocks: Optimize/Max for Sharpe Ratio
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    #Set static variables
    rfr = np.float_(0.0)
    sv = np.float_(1000000.0)

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY_norm = prices_SPY / prices_SPY.ix[0]

    df_normed = prices / prices.ix[0]

    # Finds the optimal allocations for portfolio
    def optim_SR(x, df, rfr, sv):
        df_alloc = df * x
        df_pos_vals = df_alloc * sv
        port_val = df_pos_vals.sum(axis=1) #series will symbols portfolio value
        daily_port_rets = ((port_val / port_val.shift(1)) - 1)[1:]
        sr = (np.mean(daily_port_rets - rfr) / daily_port_rets.std()) * np.sqrt(252)
        return (sr * -1) #objective function

    n = len(df_normed.columns)
    a0 = np.ones([n])/n #set number of stock allocation equal to each other
    c = ({ 'type': 'eq', 'fun': lambda a0: 1. - np.sum(a0) })
    b = [(np.float_(0.0),np.float_(1.0)) for i in range(n)] #create constraint where all allocations btw 0,1
    optim_allocs = spo.minimize(optim_SR, a0, args=(df_normed,rfr, sv), bounds=b, method='SLSQP',\
                          constraints=c)

    # Call function to Compute portfolio statistics
    cr, adr, sddr, sr = compute_porfolio_stats(prices_all, allocs=list(optim_allocs.x), sv=sv, rfr=rfr)

    # Calculate daily portfolio values with optimized allocations
    df_alloc = df_normed * list(optim_allocs.x)
    df_pos_vals = df_alloc * sv
    port_val = df_pos_vals.sum(axis=1) #series will symbols portfolio value
    port_val_norm = port_val / port_val.ix[0]

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val_norm, prices_SPY_norm], keys=['Portfolio', 'SPY'], axis=1)
        ax = df_temp.plot(title = 'Daily Porfolio Value and SPY', grid=True)
        fig = ax.get_figure()
        fig.savefig('output/comparison_optimal.png')

    return list(optim_allocs.x), cr, adr, sddr, sr

def compute_porfolio_stats(in_prices_all, allocs, sv=1000000, rfr=0.0, sf=252.0):

    #created default evenly distributed weights
    if allocs is None:
        allocs = np.ones([len(in_prices_all.columns)])/len(in_prices_all.columns)

    prices = in_prices_all.drop(['SPY'], axis=1, inplace=False)  # only portfolio symbols
    prices_SPY = in_prices_all['SPY']   # only SPY, for comparison later
    prices_SPY_norm = prices_SPY / prices_SPY.ix[0]

    # Get daily portfolio value
    # Calculates norm portfolio value for all but SPY
    df_normed = prices / prices.ix[0]
    df_alloc = df_normed * allocs
    df_pos_vals = df_alloc * sv
    port_val = df_pos_vals.sum(axis=1) #series will symbols portfolio value
    port_val_norm = port_val / port_val.ix[0]

    # Get portfolio statistics (note: std_daily_ret = volatility)
    # Calculate - daily returns without first row
    daily_port_rets = ((port_val / port_val.shift(1)) - 1)[1:]

    # Calculate - cumulative returns: value of portfolio from beginning to the end
    cr = (port_val[-1] / port_val[0]) -1

    # Calculate - average daily returns
    adr = daily_port_rets.mean()

    # Calculate -- standard deviation of average return
    sddr = daily_port_rets.std()

    #caculate -- sharpe ratio
    sr = (np.mean(daily_port_rets - rfr) / daily_port_rets.std()) * np.sqrt(252)


    # EV not used for this assignment but kept calc in code
    ev = sv * (1-cr)

    return cr, adr, sddr, sr

if __name__ == "__main__":
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,12,31)
    #symbols = ['GOOG', 'AAPL', 'GLD', 'XOM'] #example 1
    #symbols = ['AXP', 'HPQ', 'IBM', 'HNZ'] #example 2
    #symbols = ['YHOO', 'HPQ', 'GLD', 'HNZ'] #example 4
    #symbols = ['YHOO', 'XOM', 'GLD', 'HNZ']
    symbols = ['IBM', 'AAPL', 'HNZ', 'XOM', 'GLD'] #hand in example

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr
