"""
Strategy Learner
Leverages Qlearner to output orders for a particular stock

"""

import numpy as np
import csv
from util import get_data
import datetime as dt
import pandas as pd
import random as rand

class StrategyLearner(object):

    def __init__(self, verbose=False):
        self.verbose = verbose

        ############## initialize values ##############
        self.num_actions = 3 # 0) No Action, 1) Buy, 2) Sell
        self.num_states = 10000 * 7 # number of technical indicators: 3 indicators @ discretized to 0-9 + holding state
        self.s = 0 # first state is 0 or the first trading day
        self.a = 0
        self.holding = 0 #do nothing in the first holding state

        self.trade_tbl = np.random.uniform(-1.0, 1.0, size=(self.num_states, self.num_actions)) #rewards or daily return


    def calcFeatures(self, data):
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

        # self.alpha = np.float(alpha)
        # self.gamma = np.float(gamma)
        self.symbol = symbol

        data = get_data([symbol], pd.date_range(sd, ed)) #returns symbol with closing prices

        df = self.calcFeatures(data) #returns states and daily returns
        #df['yesterday'] = df[symbol].shift(1)
        #print df.head()

        # Calculate daily portfolio value
        df['price_norm'] = df[symbol] / df[symbol].ix[0]
        df['portval'] = df['price_norm'] * 100
        df['portval_yesterday'] = df['portval'].shift(1)
        df['daily_price_norm'] = df['portval'] / df['portval_yesterday']
        if self.verbose: print df.head()

        for i in range(0, len(df)):
            # calculate daily portfolio values for each state and action
            #hold = 0 #indicates not holding, 1 = long, 2 = short
            #decide how to best update trade_tbl or pandas

                # state = +0 do nothing
                n_state =  str(df.ix[i, 'state'])+'0'
                self.trade_tbl[n_state, '0'] = 0
                self.trade_tbl[n_state, '1'] = 0
                self.trade_tbl[n_state, '2'] = 0

                # state = +1 long
                b_state = str(df.ix[i, 'state'])+'1'
                self.trade_tbl[b_state, '0'] = 0
                self.trade_tbl[b_state, '1'] = -100
                self.trade_tbl[b_state, '2'] = df.ix[i, 'daily_ret']

                # state = +2 short
                s_state = str(df.ix[i, 'state'])+'2'
                self.trade_tbl[s_state, '0'] = 0
                self.trade_tbl[s_state, '1'] = df.ix[i, 'daily_ret'] * -1
                self.trade_tbl[s_state, '2'] = -100

        print self.trade_tbl

    def getAction(self, s_prime):
        """
        modify qleaner methdology to select action for trade policy --basically use argmax
        """
        # Access Trading Table to get best action per state
        s_prime_holding = str(s_prime) + str(self.holding)

        ls_returns = [self.trade_tbl[s_prime_holding, act] for act in range(0, self.trade_tbl.shape[1])]
        best_return = max(ls_returns)

        # check if there is more than one action with best_return value, if so, pick one at random
        count = ls_returns.count(best_return)
        if count > 1:
            rmax = [i for i in range(0,len(ls_returns)) if ls_returns[i] == best_return]
            action = rand.choice(rmax)
        else:
            action = np.argmax(ls_returns)

        # Update return values with a formula to discount rewards

        # 2) Choose random action with probability self.rar (TODO try with random actions)
        # if rand.random() < self.rar:
        #     action = rand.randint(0, self.num_actions-1)
        #     if self.verbose: print 'this action is random'
        # Decay Random Action Rate
        # self.rar = self.rar * self.radr # decay rar with radr

        if self.verbose:
            print "s =", self.s,"a =",action,"best_return =",best_return, "s'=", s_prime_holding
            #print self.trade_tbl

        if action == 1 and self.holding == 1:
            self.s = s_prime_holding
            self.a = action
            if action <> 0:
                self.holding = action
            return '0', s_prime_holding
        elif action == 2 and self.holding == 2:
            self.s = s_prime_holding
            self.a = action
            if action <> 0:
                self.holding = action
            return '0', s_prime_holding
        else:
            self.s = s_prime_holding
            self.a = action
            if action <> 0:
                self.holding = action
            return action, s_prime_holding


    def testPolicy(self,symbol, sd, ed, sv):
        """
        @summary: Test training set to leverage Q table built in addEvidence Method
        @param symbol: Trading Symbol
        @param sd: starting date for testing data evidence
        @param ed: ending date for testing data evidence
        @param svd: starting value for testing data evidence
        @returns: dataframe with daily actions/trades given Q-table
        """

        data = get_data([symbol], pd.date_range(sd, ed)) #returns symbol with closing prices
        df = self.calcFeatures(data) #returns states and daily returns

        for i in range(0, len(df)):
            s_next = df.ix[i, 'state']
            action, s_holding = self.getAction(s_next)
            df.ix[i, 'action'] = action
            df.ix[i, 's_holding'] = s_holding

            # backtest order file
        df['type'] = df.s_holding.apply(lambda x: x[-1:])

        if self.verbose: print df[['action', 's_holding', 'type']]

        return df[['action']]

if __name__== "__main__":
    main()