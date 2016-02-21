"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'] )
    orders.sort_index(inplace=True) #TODO check if reading in only trading dates only
    start_date = dt.datetime.strftime(orders.index.min(), '%Y-%m-%d')
    end_date = dt.datetime.strftime(orders.index.max(), '%Y-%m-%d')
    syms = list(orders.Symbol.unique())

    df_prices = get_data(syms, pd.date_range(start_date, end_date), addSPY=False)
    df_prices['cash'] = 1

    orders['share_sign'] = orders.apply(lambda x: -1.0 if x['Order'] == 'SELL' else 1.0, axis=1)
    orders['stock_price'] = orders.apply(lambda x: df_prices.loc[x.name][x.Symbol], axis=1)

    df_trades = pd.DataFrame(0, index=pd.date_range(start_date, end_date), columns= syms + ['cash'])

    #step by step of orders and catalogs in df_trades
    def log_trades(dt, o_sym, o_shares, o_sign, o_sp):
        #log into trading table on date - sell or buy # of shares
        df_trades.loc[dt][o_sym] = o_shares * o_sign
        df_trades.loc[dt]['cash'] += (o_sp * o_shares * o_sign) * -1 # TODO does not take into consideration

    #exeutes step by step row application
    tmp = orders.apply(lambda x: log_trades(dt=x.name, o_sym=x.Symbol, o_shares=x.Shares,
                                     o_sign=x.share_sign, o_sp = x.stock_price), axis=1)

    #df_holdings = accumulate asset value
    df_holdings = pd.DataFrame(0, index=pd.date_range(start_date, end_date), columns= syms + ['cash'])
    #sets the first row in holdings
    df_holdings.iloc[0] = df_trades.iloc[0]
    df_holdings['cash'][0] = start_val + df_trades['cash'][0]

    for y in range(0,len(df_holdings.columns)): # columns
        for x in range(1,len(df_holdings)):     # skips the first row
                df_holdings.ix[x, y] = df_holdings.ix[(x-1), y] + df_trades.ix[x, y]

    df_value = df_prices.multiply(df_holdings, axis='columns')
    df_portval = df_value.sum(axis=1)

    return df_portval

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders.csv"
    sv = 1000000
    rfr = 0.0
    sf=252.0

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime.strftime(portvals.index.min(), '%Y-%m-%d')
    end_date = dt.datetime.strftime(portvals.index.max(), '%Y-%m-%d')

    daily_port_rets = ((portvals / portvals.shift(1)) - 1)[1:]
    cum_ret = (portvals[-1] / portvals[0]) -1  #cumulative returns of portfolio value
    avg_daily_ret = daily_port_rets.mean()
    std_daily_ret = daily_port_rets.std()
    sharpe_ratio = (np.mean(daily_port_rets - rfr) / daily_port_rets.std()) * np.sqrt(sf)

    prices_SPY = get_data(['$SPX'], pd.date_range(start_date, end_date), addSPY=False)
    daily_port_rets_SPY = ((prices_SPY / prices_SPY.shift(1)) - 1)[1:]
    cum_ret_SPY = (prices_SPY.iloc[-1] / prices_SPY.iloc[0]) -1  #cumulative returns of portfolio value
    avg_daily_ret_SPY = daily_port_rets_SPY.mean()
    std_daily_ret_SPY = daily_port_rets_SPY.std()
    sharpe_ratio_SPY = (np.mean(daily_port_rets_SPY - rfr) / daily_port_rets_SPY.std()) * np.sqrt(sf)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
