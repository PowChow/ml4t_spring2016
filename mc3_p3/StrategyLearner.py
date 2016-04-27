"""
Strategy Learner
Leverages Qlearner to output orders for a particular stock

"""

import numpy as np
import csv
from util import get_data
import datetime as dt

class StrategyLearner(object):

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.is_values = {}
        
    def preprocess_data(self, data, sym, in_sample):
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
       data['daily_ret'] = data.apply(lambda x: (x['t+N']/x[self.symbol]) - 1.0, axis=1)
       #data['daily_ret'] = (data['t+N']/data[self.symbol]) - 1.0             # change in price in 5 days
       data['vol'] = pd.rolling_std(data['daily_ret'], window=5)   # evaluate risk and signifigance of price movement
       data['bb_upper_band'] = data['sma'] + (2*data['vol'])
       data['bb_lower_band'] = data['sma'] - (2*data['vol'])
       #data['bb'] = (data[self.symbol] - data['sma'])/((data['bb_upper_band']-data['bb_lower_band'])/2)  #range of volatility from sma
       data['bb'] = (data[self.symbol] - pd.rolling_mean(data[self.self.symbolbol], window=5))\
                  /(2* pd.rolling_std(data[self.symbol], window=5) )  #range of volatility from sma
    
       if in_sample:
           self.is_values['momentum_mean'] = np.float(data['momentum'].mean())
           self.is_values['momentum_std'] = np.float(data['momentum'].std())
           self.is_values['sma_mean'] = np.float(data['sma'].mean())
           self.is_values['sma_std'] = np.float(data['sma'].std())
           self.is_values['vol_mean'] = np.float(data['vol'].mean())
           self.is_values['vol_std'] = np.float(data['vol'].std())
           self.is_values['bb_mean'] = np.float(data['bb'].mean())
           self.is_values['bb_std'] = np.float(data['bb'].std())
           self.is_values['price_mean'] = np.float(data[self.self.symbolbol].mean())
           self.is_values['price_std'] = np.float(data[self.self.symbolbol].std())
    
       else:
           pass
    
    
       data['momentum_norm'] = (data['momentum'] - is_values['momentum_mean'])/is_values['momentum_std']
       data['sma_norm'] = (data['sma'] - is_values['sma_mean'])/is_values['sma_std']
       data['vol_norm'] = (data['vol'] - is_values['vol_mean'])/is_values['vol_std']
       data['bb_norm'] = data['bb']
       data['price_norm'] =(data[self.symbol] - is_values['price_mean'])/is_values['price_std']
    
       data.dropna(how='any', inplace=True)
    
       # printing for debugging
       #print data.head(10)
       #print data.describe()
    
        return data     

    def addEvidence(self,symbol, sd, ed, sv, alpha=0.2, gamma=0.9):
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
        self.num_states = 10000 # number of technical indicators: 3 indicators @ discretized to 0-9 + holding state

        self.Q_tbl = np.random.uniform(-1.0, 1.0,
                                       size=(self.num_states, self.num_actions)) #rewards or daily return

        self.s = 0 # first state is 0 or the first trading day
        self.a = 0 # first action is nothing
        self.hold = 0 #indicates not holding, 1 = long, 2 = short 
        self.alpha = np.float(alpha)
        self.gamma = np.float(gamma)
        self.symbol = symbol

        data = get_data(list(symbol), pd.date_range(sd, ed)) #returns symbol with closing prices

        tech_data = preprocess(data, )
        #discretize indicators
    
        for d in data.index:
        # calculate technical features for Q value table + results from selected actions
        # discretize indicators
        # update Q table which are values of daily returns
        # dataframe to collect daily trading action / states -- output to debug or to run in the simulator
        # reference holding state to figure out how to ouput order signal

    def query(self, ,symbol, sd, ed, sv):
        """
        @summary: Test training set to leverage Q table built in addEvidence Method
        @param symbol: Trading Symbol
        @param sd: starting date for testing data evidence
        @param ed: ending date for testing data evidence
        @param svd: starting value for testing data evidence
        @returns: dataframe with daily actions/trades given Q-table
        """
        arrayX = self.X
        arrayY = self.Y
        attributes = self.X.shape[1]


        #TODO add code to run through states and access Q table, will we need to update Q table here or just used what we learned?

        return data_prices

if __name__== "__main__":
    main()