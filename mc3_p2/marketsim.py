"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data

def compute_portvals(orders_file = "./Orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code

    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'] )
    orders.sort_index(inplace=True)
    start_date = dt.datetime.strftime(orders.index.min(), '%Y-%m-%d')
    end_date = dt.datetime.strftime(orders.index.max(), '%Y-%m-%d')


    #################################################################################################
    def execute_orders(in_orders, in_start_date, in_end_date):
        syms = list(in_orders.Symbol.unique())

        df_prices = get_data(syms, pd.date_range(in_start_date, in_end_date), addSPY=True)
        df_prices.drop(['SPY'], axis=1, inplace=True) #drop index prices after using it to get trading only days
        df_prices['cash'] = 1
        df_copy = df_prices #use as a place holder because it messages with prices

        in_orders['share_sign'] = in_orders.apply(lambda x: -1.0 if x['Order'] == 'SELL' else 1.0, axis=1)
        in_orders['stock_price'] = in_orders.apply(lambda x: df_prices.loc[x.name][x.Symbol], axis=1)

        #dataframe logs trades by date
        df_trades = df_copy.copy(deep=False)
        df_trades.ix[:] = 0

        #step by step of orders and catalogs in df_trades
        #this function can be reiterated when checking for leverage
        def log_trades(dt, o_sym, o_shares, o_sign, o_sp):
            #log into trading table on date - sell or buy # of shares
            df_trades.loc[dt][o_sym] += o_shares * o_sign
            df_trades.loc[dt]['cash'] += (o_sp * o_shares * o_sign) * -1

        #calculates df_trades
        # exeutes step by step row Order execution
        tmp = in_orders.apply(lambda x: log_trades(dt=x.name, o_sym=x.Symbol, o_shares=x.Shares,
                                         o_sign=x.share_sign, o_sp = x.stock_price), axis=1)

        #calculates holdings
        df_holdings = df_copy.copy(deep=False)
        df_holdings[:] = 0

        #sets the first row in holdings
        df_holdings.iloc[0] = df_trades.iloc[0]
        df_holdings['cash'][0] = start_val + df_trades['cash'][0]

        df_leverage = pd.DataFrame(0, index=df_copy.index, columns=['lev'])

        for x in range(1,len(df_holdings)):     # skips the first row
            for y in range(0,len(df_holdings.columns)): # columns
                df_holdings.ix[x, y] = df_holdings.ix[(x-1), y] + df_trades.ix[x, y]

        for row in range(0, len(df_holdings)): #by row
            row_pos = 0.0
            row_neg = 0.0
            cash = 0.0
            for col in range(0, len(df_holdings.columns)-1): #by columns
                if df_holdings.iloc[row][col] >= 0:
                    row_pos += df_holdings.iloc[row][col] * df_prices.loc[df_holdings.index[row]][df_holdings.columns[col]]
                elif df_holdings.iloc[row][col] < 0:
                    row_neg += df_holdings.iloc[row][col] * df_prices.loc[df_holdings.index[row]][df_holdings.columns[col]]
            cash += df_holdings.iloc[row]['cash']
            df_leverage.iloc[row] = (row_pos + abs(row_neg)) / ((row_pos - abs(row_neg)) + cash)
            #print df_leverage.iloc[row]


        df_prices['cash']=1
        df_value = df_prices.multiply(df_holdings, axis='columns')
        df_portval = df_value.sum(axis=1)
        df_portval.dropna(how='any', inplace=True)

        print df_portval.head()
        return df_leverage, df_portval
    #################################################################################################

    leverage, portval = execute_orders(orders, start_date, end_date)
    # Check leverage - Recalculate trades and holdings
    orders2 = orders
    for l in range(0, len(leverage)):
        over = 0
        # Executes step by step row Order execution
        if leverage.iloc[l]['lev'] > 2.0:
            over += 1
            orders2 = orders2.loc[orders2.index != leverage.index[l]]  # Cancel the trade from the orders
            #print 'order exceeded'

    leverage2, portval2 = execute_orders(orders2, start_date, end_date)

    return portval2

def sims_output(sv=1000000, of= "./Orders/orders.csv", gen_plot=True, symbol='ML4T-220', strat_name="KNN"):

    rfr = 0.0
    sf = 252.0

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]# just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    start_date = dt.datetime.strftime(portvals.index.min(), '%Y-%m-%d')
    end_date = dt.datetime.strftime(portvals.index.max(), '%Y-%m-%d')

    # Portfolio normalizations
    daily_port_rets = ((portvals / portvals.shift(1)) - 1)[1:]
    portval_norm = portvals / portvals.ix[0]
    cum_ret = (portvals[-1] / portvals[0]) -1  #cumulative returns of portfolio value
    avg_daily_ret = daily_port_rets.mean()
    std_daily_ret = daily_port_rets.std()
    sharpe_ratio = (np.mean(daily_port_rets - rfr) / daily_port_rets.std()) * np.sqrt(sf)

    #symbol normalization, i.e. ML4T-220 or IBM
    prices_sym = get_data([symbol], pd.date_range(start_date, end_date), addSPY=False)
    prices_sym_norm = prices_sym / prices_sym.ix[0]
    daily_port_rets_sym = ((prices_sym / prices_sym.shift(1)) - 1)[1:]
    cum_ret_sym = (prices_sym.iloc[-1] / prices_sym.iloc[0]) -1  #cumulative returns of portfolio value
    avg_daily_ret_sym = daily_port_rets_sym.mean()
    std_daily_ret_sym = daily_port_rets_sym.std()
    sharpe_ratio_sym = (np.mean(daily_port_rets_sym - rfr) / daily_port_rets_sym.std()) * np.sqrt(sf)

    # Compare portfolio against symbol
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of Symbol: {}".format(sharpe_ratio_sym)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of Symbol: {}".format(cum_ret_sym)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of Symbol: {}".format(std_daily_ret_sym)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of Symbol: {}".format(avg_daily_ret_sym)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    if gen_plot:
        #Plots comparison between Portfolio Values from Orders and symbol price
        df_temp = pd.concat([portval_norm, prices_sym_norm], keys=['Portfolio', symbol], axis=1)
        df_temp.dropna(inplace=True)
        ax = df_temp.plot(title = 'Back Test Portfolio and Symbol Prices: %s' % strat_name, grid=False)
        ax.legend(loc='upper left', labels=['Portfolio', symbol])
        fig = ax.get_figure()
        fig.savefig('./Output/%s_comparison_chart.png' % strat_name)


if __name__ == "__main__":
    run()
