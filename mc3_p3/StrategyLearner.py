"""
Strategy Learner
Leverages Qlearner to output orders for a particular stock

"""

import numpy as np
import csv
from util import get_data
import datetime as dt
import pandas as pd

class StrategyLearner(object):

    def __init__(self, verbose=False):
        self.verbose = verbose

    def calc_tech(self, data):
        """
        function pulls data set for particular homework or trading symbol
    
        :param sym: calculate technical values for qlearner states
        :param sdate: start date for data set
        :param edate: end date for data set
        :param is_values: in_sample technical values dictionary for normalization
        :return: dataframe with technical features and normalizations of those features
        """
        #calculate technical features within this dataframe - adjusted close / SMA, Bollinger Band value,return since entry

        data.drop(['SPY'], axis=1, inplace=True)
        data['t-N'] = data[self.symbol].shift(5)   # shifts days forward, look backwards 5 days
        data['t+N'] = data[self.symbol].shift(-5)  # shifts days backwards, look forwards 5 days
        data['t+1'] = data[self.symbol].shift(-1)  # shifts forwards, looks backwards 1 day
    
        # calculate technical features
        data['momentum'] = data.apply(lambda x: (x[self.symbol]/x['t-N']) - 1.0, axis=1)
        data['sma'] = pd.rolling_mean(data[self.symbol], window=5)          # define current direction with a lag - smooth price action and filter out nois
        data['ac_sma'] = data[self.symbol] / data['sma']
        data['daily_ret'] = data.apply(lambda x: (x['t+N']/x[self.symbol]) - 1.0, axis=1)
        #data['daily_ret'] = (data['t+N']/data[self.symbol]) - 1.0             # change in price in 5 days
        data['vol'] = pd.rolling_std(data['daily_ret'], window=5)            # evaluate risk and signifigance of price movement
        data['bb_upper_band'] = data['sma'] + (2*data['vol'])
        data['bb_lower_band'] = data['sma'] - (2*data['vol'])
        #data['bb'] = (data[self.symbol] - data['sma'])/((data['bb_upper_band']-data['bb_lower_band'])/2)  #range of volatility from sma
        data['bb'] = (data[self.symbol] - pd.rolling_mean(data[self.symbol], window=5)) / (2* pd.rolling_std(data[self.symbol], window=5))
        data.dropna(inplace=True)

        df_tech = data[[self.symbol, 'ac_sma', 'bb', 'momentum', 'daily_ret']]
        #discretize states
        df_tech['ac_sma'] = pd.qcut(df_tech['ac_sma'], 10, labels=['0', '1','2','3','4','5','6','7','8','9'])
        df_tech['momentum'] = pd.qcut(df_tech['momentum'], 10, labels=['0', '1','2','3','4','5','6','7','8','9'])
        df_tech['bb'] = pd.qcut(df_tech['bb'], 10, labels=['0','1','2','3','4','5','6','7','8','9'])

        #df_tech['combined'] = df_tech.ac_sma.map('str') + df_tech.momentum.map('str') + df_tech.bb.map('str')
        df_tech['state'] = df_tech.apply(lambda x:'%s%s%s' % (x['ac_sma'],x['bb'],x['momentum']),axis=1)

        # printing for debugging
        #print df_tech[[self.symbol, 'daily_ret', 'state']].head()
        #print data.describe()
    
        return df_tech[[self.symbol, 'daily_ret', 'state']]

    def compute_portvals(prices, sv=1000000, rfr=0.0, sf=252.0):

        #created default evenly distributed weights
        if allocs is None:
            allocs = np.ones([len(prices.columns)])/len(prices.columns)

        # Get daily portfolio value
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


    def addEvidence(self, symbol, sd, ed, sv, alpha=0.2, gamma=0.9):
        """
        @summary: Add training data to learner, append to existing
        @param symbol: Trading Symbol
        @param sd: starting date for training data evidence
        @param ed: ending date for training data evidence
        @param svd: starting value for training data evidence
        @returns: ?? Adds vales to Q table that will be used in testing and trades

        """
        ############## initialize values ##############
        self.num_actions = 3 # 0) No Action, 1) Buy, 2) Sell
        self.num_states = 10000 * 3 # number of technical indicators: 3 indicators @ discretized to 0-9 + holding state


        self.trade_tbl = np.random.uniform(-1.0, 1.0,
                                       size=(self.num_states, self.num_actions)) #rewards or daily return

        self.s = 0 # first state is 0 or the first trading day
        self.a = 0 # first action is nothing
        self.alpha = np.float(alpha)
        self.gamma = np.float(gamma)
        self.symbol = symbol

        data = get_data([symbol], pd.date_range(sd, ed)) #returns symbol with closing prices

        df_tech = self.calc_tech(data) #returns states and daily returns
        #df_tech['yesterday'] = df_tech[symbol].shift(1)
        #print df_tech.head()

        # Calculate daily portfolio value
        df_tech['price_norm'] = df_tech[symbol] / df_tech[symbol].ix[0]
        df_tech['portval'] = df_tech['price_norm'] * sv
        df_tech['portval_yesterday'] = df_tech['portval'].shift(1)
        df_tech['daily_price_norm'] = df_tech['portval'] / df_tech['portval_yesterday']
        print df_tech.head()

        for i in range(0, len(df_tech)):
            pass
            # calculate daily portfolio values for each state and action
            #hold = 0 #indicates not holding, 1 = long, 2 = short
            #decide how to best update trade_tbl or pandas
                # state = +0 do nothing
                # self.trade_tbl[str(df[i, 'state'])+'0', '0'] =
                # self.trade_tbl[str(df[i, 'state'])+'0', '1'] =
                # self.trade_tbl[str(df[i, 'state'])+'0', '2'] =

                # state = +1 buy
                # self.trade_tbl[str(df[i, 'state'])+'0', '0'] =
                # self.trade_tbl[str(df[i, 'state'])+'0', '1'] =
                # self.trade_tbl[str(df[i, 'state'])+'0', '2'] =

                # state = +2 sell
                # self.trade_tbl[str(df[i, 'state'])+'0', '0'] =
                # self.trade_tbl[str(df[i, 'state'])+'0', '1'] =
                # self.trade_tbl[str(df[i, 'state'])+'0', '2'] =

    def testPolicy(self,symbol, sd, ed, sv):
        """
        @summary: Test training set to leverage Q table built in addEvidence Method
        @param symbol: Trading Symbol
        @param sd: starting date for testing data evidence
        @param ed: ending date for testing data evidence
        @param svd: starting value for testing data evidence
        @returns: dataframe with daily actions/trades given Q-table
        """

        #TODO add code to run through states and access Q table, will we need to update Q table here or just used what we learned?
        #TODO create output of orders for backtesting
        # run through test data by date
        # calculate ongoing values
        # accessing trading table to make decisions when states are similar
        # make decision based on max trade table value
        # OUTPUT action to data frame == orders file
        # backtest order file

        return 'hello'

if __name__== "__main__":
    main()