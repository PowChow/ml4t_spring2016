"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import datetime as dt

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']   # only SPY, for comparison later
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

    #Default statement to run code
    #cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1]

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # TODO add code to plot here
        df_temp = pd.concat([port_val_norm, prices_SPY_norm], keys=['Portfolio', 'SPY'], axis=1)
        ax = df_temp.plot(title ='Daily Portfolio Value and SPY', grid=True)
        fig = ax.get_figure()
        fig.savefig('output/plot.png')

    # TODO Add code here to properly compute end value
    ev = sv * (1-cr)

    return cr, adr, sddr, sr, ev

if __name__ == "__main__":
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2010, 12, 31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM'] #use this one for graded plot
    #symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    #allocations = [0.2, 0.3, 0.4, 0.1]
    allocations = [0.2, 0.2, 0.4, 0.2] # use this for grading
    #allocations = [0.0, 0.0, 0.0, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date.strftime('%Y-%m-%d')
    print "End Date:", end_date.strftime('%Y-%m-%d')
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio (calc):", sr
    print "Volatility (stdev of daily returns)(calc):", sddr
    print "Average Daily Return (calc):", adr
    print "Cumulative Return (calc):", cr
    print "Ending Value of Portfolio (calc):", ev
