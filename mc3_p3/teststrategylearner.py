import StrategyLearner as sl
import datetime as dt
import csv
from marketsim import sims_output, compute_portvals
import pandas as pd
import time


# run the code to test a learner
if __name__=="__main__":

    verbose = False
    symbol = 'IBM'

    for iteration in range(0,1):

        learner = sl.StrategyLearner(verbose = verbose) # constructor

        learner.addEvidence(symbol = "IBM", sd=dt.datetime(2008,1,1),
                            ed=dt.datetime(2009,1,1), sv = 10000) # training step

        df_trades = learner.testPolicy(symbol = "IBM", sd=dt.datetime(2009,1,1),
                                       ed=dt.datetime(2010,1,1), sv = 10000) # testing step
        #df_trades.reset_index(inplace=True)
        df_trades.dropna(inplace=True)
        #print df_trades

        if verbose: print iteration

        with open('./Orders/%s_orders.csv' % symbol, 'w+') as csvfile:
            fieldnames = ['Date', 'Symbol', 'Order', 'Shares']

            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(fieldnames)

            for i in range(0, len(df_trades)):

                trade_dt = df_trades.index[i].strftime('%Y-%m-%d')


                if df_trades.ix[i, 'action'] == 1:
                    # Date, Symbol, Order, Shares
                    writer.writerow([trade_dt, symbol, 'BUY', 100])
                elif df_trades.ix[i, 'action'] == 2:
                    writer.writerow([trade_dt, symbol, 'SELL', 100])
                else:
                    pass

    # 5) Run orders through market simulators and output back testing graph
    sims_output(sv=10000, of='./Orders/%s_orders.csv' % symbol, gen_plot=True,
                 symbol=symbol, strat_name='Strategy_%s'% (symbol))