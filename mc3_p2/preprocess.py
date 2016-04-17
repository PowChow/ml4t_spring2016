"""
1) Preprocess File for MC3 P2
KNN Stock Return Predictor and Trading Strategy

"""
import numpy as np
import math
import time
import datetime as dt
import pandas as pd
import csv
from pandas.tseries.offsets import BDay


import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import LinRegLearner as lrl
import BagLearner as bl
import KNNLearner as knn
from util import get_data
from marketsim import sims_output, compute_portvals

######################################################
############## PLOTTING #############################
######################################################

def plot_lines_data(price_norm, actualY, predY, name='default'):
    """
    Plots three normalized values: normalized price, actual Y, and predicted Y
    Outputs the graph as PNG
    """
    ax = price_norm.plot(title=name, label='Norm Price')
    actualY.plot(label='Actual Price', color='green', ax=ax)
    predY.plot(label='Predicted Price', color='crimson', ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Values")
    ax.legend(loc='lower left', labels=['Price Norm', 'Actual Y', 'Predicted Y'])

    print 'Actual Y stats: ', actualY.describe()
    print 'Predicted Y stats: ', predY.describe()

    #plt.show()
    fig = ax.get_figure()
    fig.savefig('./Output/%s_lines.png' % (name))


def plot_strategy(price, of='./Orders/orders.csv', name='default'):
    """
    Plots Symbol Price Data with Entry and Exit for
    Short and Long Trading Strategies based on Learner
    """
    #print price.head(10)

    orders = pd.read_csv(of)
    ax = price.plot(title=name, label='Price')
    ax.set_xlabel("Date")
    ax.set_ylabel("Prices")
    ax.legend().set_visible(False)


    # Plot vertical lines for shorts
    for i in range(0, len(orders)):
        if (orders.ix[i]['Strat'] == 'Short') and (orders.ix[i]['Type'] == 'Exit'):
            ax.axvline(orders.ix[i]['Date'], color='r', linestyle='solid')
        elif (orders.ix[i]['Strat'] == 'Short') and (orders.ix[i]['Type'] == 'Entry'):
            ax.axvline(orders.ix[i]['Date'], color='black', linestyle='solid')
        elif (orders.ix[i]['Strat'] == 'Long') and (orders.ix[i]['Type'] == 'Entry'):
            ax.axvline(orders.ix[i]['Date'], color='green', linestyle='solid')
        elif (orders.ix[i]['Strat'] == 'Long') and (orders.ix[i]['Type'] == 'Exit'):
            ax.axvline(orders.ix[i]['Date'], color='black', linestyle='solid')
        else:
            pass

    plt.show()
    fig = ax.get_figure()
    fig.savefig('./Output/%s_strategy_exitentry.png' % (name))

def plot_backtest():
    pass


#########################################################
############## CALCULATIONS #############################
#########################################################

def preprocess_data(sym, sdate, edate, in_sample=False, is_values={}):
    """
    function pulls data set for particular homework or trading symbol

    :param sym: trading or homework sym to calculate technical
    :param sdate: start date for data set
    :param edate: end date for data set
    :param is_values: in_sample technical values dictionary
    :return:dataframe with technical features and normalizations of those features
    """

    df = get_data(sym, pd.date_range(sdate, edate)) #returns symbol with closing prices
    df.drop(['SPY'], axis=1, inplace=True)
    df['t-N'] = df[sym].shift(5)   # shifts days forward, look backwards 5 days
    df['t+N'] = df[sym].shift(-5)  # shifts days backwards, look forwards 5 days
    df['t+1'] = df[sym].shift(-1)  # shifts forwards, looks backwards 1 day

    # calculate technical features
    df['momentum'] = (df[sym]/df['t-N']) - 1.0              # measures the rise and fall of stock, strength and weakness of current price
    df['sma'] = pd.rolling_mean(df[sym], window=5)          # define current direction with a lag - smooth price action and filter out noise
    df['daily_ret'] = (df['t+N']/df[sym]) - 1.0             # change in price in 5 days
    df['vol'] = pd.rolling_std(df['daily_ret'], window=5)   # amont of variability or dispersion around the average -- evaluate risk and signifigance of price movement
    df['bb_upper_band'] = df['sma'] + (2*df['vol'])
    df['bb_lower_band'] = df['sma'] - (2*df['vol'])
    #df['bb'] = (df[sym] - df['sma'])/((df['bb_upper_band']-df['bb_lower_band'])/2)  #range of volatility from sma
    df['bb'] = (df[sym] - pd.rolling_mean(df[sym], window=5))\
               /(2* pd.rolling_std(df[sym], window=5) )  #range of volatility from sma


    if in_sample:
        is_values = {}
        is_values['momentum_mean'] = np.float(df['momentum'].mean())
        is_values['momentum_std'] = np.float(df['momentum'].std())
        is_values['sma_mean'] = np.float(df['sma'].mean())
        is_values['sma_std'] = np.float(df['sma'].std())
        is_values['vol_mean'] = np.float(df['vol'].mean())
        is_values['vol_std'] = np.float(df['vol'].std())
        is_values['bb_mean'] = np.float(df['bb'].mean())
        is_values['bb_std'] = np.float(df['bb'].std())
        is_values['price_mean'] = np.float(df[sym].mean())
        is_values['price_std'] = np.float(df[sym].std())

    else:
        pass


    df['momentum_norm'] = (df['momentum'] - is_values['momentum_mean'])/is_values['momentum_std']
    df['sma_norm'] = (df['sma'] - is_values['sma_mean'])/is_values['sma_std']
    df['vol_norm'] = (df['vol'] - is_values['vol_mean'])/is_values['vol_std']
    df['bb_norm'] = df['bb']
    df['price_norm'] =(df[sym] - is_values['price_mean'])/is_values['price_std']

    df.dropna(how='any', inplace=True)

    # printing for debugging
    print df.head(10)
    print df.describe()

    if in_sample:
        return df, is_values
    else:
        return df


def SendtoModel(train_df, train_price, test_df, test_price, model='knn', symbol='IBM', k=3, bags=0, verbose=False):
    """
    Sends test and train data frame to selected model
    Returns predicted test Y and train Y
    """

    #calculate training and test sets
    trainX = np.array(train_df.iloc[0:,0:-1])
    trainY = np.array(train_df.iloc[0:,-1])
    testX = np.array(test_df.iloc[0:,0:-1])
    testY = np.array(test_df.iloc[0:,-1])

    print 'shape testX', testX.shape
    print 'shape testY', testY.shape

    if model == 'knn':
        learner = knn.KNNLearner(k=k, verbose=True) # create a knnLearner
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY_train = learner.query(trainX) # get the predictions
        rmse_train = math.sqrt(((trainY - predY_train) ** 2).sum()/trainY.shape[0])

        # evaluate out of sample
        predY_test = learner.query(testX) # get the predictions
        rmse_test = math.sqrt(((testY - predY_test) ** 2).sum()/testY.shape[0])

        #output graphs - normalized lines
        # plot_lines_data(price_norm=train_price, actualY=train_df.iloc[0:,-1], predY=pd.Series(predY_train, index=train_df.index),
        #           name='%s_in_sample_%s' % (symbol[0], model))
        # plot_lines_data(price_norm=test_price, actualY=test_df.iloc[0:,-1], predY=pd.Series(predY_test, index=test_df.index),
        #           name='%s_out_sample_%s' % (symbol[0], model))


        if verbose:
            #(a) in sample results
            print model, 'with arguments k=%s, bags=%s' % (k, bags)
            print "In sample results"
            print "RMSE: ", rmse_train
            c = np.corrcoef(predY_train, y=trainY)
            print "corr: ", c[0,1]

            #(b) out of sample results
            print
            print model, 'with arguments k=%s, bags=%s' % (k, bags)
            print "Out of sample results"
            print "RMSE: ", rmse_test
            c = np.corrcoef(predY_test, y=testY)
            print "corr: ", c[0,1]
            print 'length of predicted values: ', len(predY_test)
            #print 'print predicted Y values:', predY_test

        else:
            pass

        return predY_train, predY_test

def create_rolling_orders(predY_df, sym='IBM'):
    """
    :param: pred_returns set of predicted returns with dates
    :param: sym generate orders, buys and sells, for symbol
    returns: outputs list of orders, returns nothing
    """
    #TODO check overlap of orders with rolling method

    with open('./Orders/%s_knn_orders_rolling.csv' % sym, 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for i, row in predY_df.iterrows():

        # Orders Output -- trading policies or strategies
            curr_date = dt.datetime.strftime(i, '%Y-%m-%d')
            trade_date = i - pd.Timedelta(days=4)
            trade_date = dt.datetime.strftime(trade_date, '%Y-%m-%d')

            if row[0] >= .01:
                # Date, Symbol, Order, Shares
                writer.writerow([trade_date, sym[0], 'BUY', 100])
                writer.writerow([curr_date, sym[0], 'SELL', 100])
            elif row[0] <= -.01:
                writer.writerow([trade_date, sym[0], 'SELL', 100])
                writer.writerow([curr_date, sym[0], 'BUY', 100])
            else:
                pass

def create_5day_orders(df, sym='IBM', type='insample'):
    """
    :param: pred_returns set of predicted returns with dates
    :param: sym generate orders, buys and sells, for symbol
    returns: outputs list of orders, returns nothing
    """
    #returns_only = df['predY_returns'].dropna(how='any', inplace=False)

    with open('./Orders/%s_knn_orders_5day_%s.csv' % (sym[0], type ), 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Date','Symbol','Order','Shares', 'Type', 'Strat'])

        for i in xrange(0, len(df), 5):
            # Orders Output -- trading policies or strategies
            curr_date = dt.datetime.strftime(df.index[i], '%Y-%m-%d')
            trade_date = df.index[i-4]
            trade_date = dt.datetime.strftime(trade_date, '%Y-%m-%d')
            #print curr_date, trade_date

            if df['predY_returns'][i] >= .01:
                # Date, Symbol, Order, Shares
                writer.writerow([trade_date, sym[0], 'BUY', 100, 'Entry', 'Long'])
                writer.writerow([curr_date, sym[0], 'SELL', 100, 'Exit', 'Long'])
            elif df['predY_returns'][i] <= -.01:
                writer.writerow([trade_date, sym[0], 'SELL', 100, 'Entry','Short'])
                writer.writerow([curr_date, sym[0], 'BUY', 100, 'Exit', 'Short'])
            else:
                pass


######################################################
############## EXECUTION #############################
######################################################

if __name__ == "__main__":
    # 1) Set constants
    start_val = 1000000
    in_sample_dict = {}  # dictionary to hold in sample technical values stats

    # 2) Get and process training set
    #sym = ['IBM']
    sym = ['ML4T-220']
    is_start_dt = dt.datetime(2007, 12, 31)
    is_end_dt = dt.datetime(2009, 12, 31)
    is_df = get_data(sym, pd.date_range(is_start_dt, is_end_dt)) #returns symbol with closing prices
    is_spy_df = is_df['SPY']
    is_mean = np.float(is_df[sym].mean())
    is_sd = np.float(is_df[sym].std())

    #A ######TRAINING DATA
    train_data, in_sample_dict = preprocess_data(sym=sym, sdate=is_start_dt, edate=is_end_dt, in_sample=True)

    train = train_data[['momentum_norm', 'sma_norm', 'vol_norm', 'bb', 'daily_ret']]
    train.to_csv('Data/%s_train_norms.csv' % sym[0], index=False, encoding='utf-8', header=False)

    #B ######TESTING DATA
    oos_df = get_data(sym, pd.date_range(dt.datetime(2009, 12, 31), dt.datetime(2011, 12, 31))) #returns symbol with closing prices
    oos_spy_df = oos_df['SPY']
    test_data = preprocess_data(sym=sym, sdate=dt.datetime(2009, 12, 31), edate=dt.datetime(2011, 12, 31),
                                in_sample=False, is_values=in_sample_dict)
    test = test_data[['momentum_norm', 'sma_norm', 'vol_norm', 'bb', 'daily_ret']]
    test.to_csv('Data/%s_test_norms.csv' % sym[0],index=False,encoding='utf-8', header=False)

    # 3) Send test and train data to model
    pred_train_Y, pred_test_Y = SendtoModel(train_df=train, train_price=train_data['price_norm'],
                                    test_df=test, test_price=test_data['price_norm'],
                                    model='knn', symbol=sym, verbose=True, k=3)


    # 4) Create orders from predictions by merging predicted Y-returns with SPY-dates
    # a) in sample
    returns_train_df = pd.concat([is_spy_df, pd.DataFrame(pred_train_Y, index = train.index, columns=['predY_returns'])], axis=1)
    create_5day_orders(returns_train_df, sym=sym, type='insample')
    # plot_strategy(price=is_df[sym], of='./Orders/ML4T-220_knn_orders_5day_insample.csv', name='%s_in_sample' % sym[0])


    # b) out of sample
    returns_test_df = pd.concat([oos_spy_df, pd.DataFrame(pred_test_Y, index = test.index, columns=['predY_returns'])], axis=1)
    create_5day_orders(returns_test_df, sym=sym, type='outsample')
    #plot_strategy(price=oos_df[sym], of='./Orders/ML4T-220_knn_orders_5day_outsample.csv', name='%s_out_sample' % sym[0])

    #create_rolling_orders(pred_return_df, sym=sym)  #TODO extra credit


    # 5) Run orders through market simulators and output back testing graph
    sims_output(sv=start_val, of='./Orders/%s_knn_orders_5day_%s.csv' % (sym[0], 'insample'),
                gen_plot=True, strat_name='5day_KNN_%s_%s'% (sym[0], 'insample'))

    # sims_output(sv=start_val, of='./Orders/%s_knn_orders_5day_%s.csv' % (sym, 'insample'), symbol=sym,
    #             gen_plot=False, strat_name='5day_KNN')


    # 6) output first graphs or is this task embedded in other areas?
    # predicts 5 days ahead, and always closes its position 5 days after opening it:
    # extra credit -- rolling variation of KNN, add a feature in KNNlearner with rolling=True



#market sim
# create function to create csv of orders for back testing
# send to trading simulator / back testing
# calcuate 5-day relative return, Y


# send testing set to test predictions

#output graph


