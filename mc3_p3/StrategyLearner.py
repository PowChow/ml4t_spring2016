"""
Strategy Learner
Leverages Qlearner to output orders for a particular stock

"""

import numpy as np
import math
import csv
from scipy.spatial import distance

class StrategyLearner(object):

    def __init__(self, verbose=False):
        self.verbose = verbose


    learner.addEvidence(symbol = "IBM", sd=dt.datetime(2008,1,1),
                        ed=dt.datetime(2009,1,1), sv = 10000) # training step

    def addEvidence(self,symbol, sd, ed, sv):
        """
        @summary: Add training data to learner, append to existing
        @param symbol: Trading Symbol
        @param sd: starting date for training data evidence
        @param ed: ending date for training data evidence
        @param svd: starting value for training data evidence

        """
        self.
        
    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built
        @param points: should be a numpy array with each row corresponding to a specific query
        @returns average Y according to the saved model
        """
        arrayX = self.X
        arrayY = self.Y
        attributes = self.X.shape[1]

        #create zero array with size of points for Y predicted values
        predY = []

        # for loop to calculate Euclidean distance between query point and all
        print len(points)
        for p in points:
            d = distance.cdist(arrayX, np.reshape(p, newshape=(1,attributes)), metric='euclidean')
            #d = distance.cdist(arrayX, p, metric='euclidean')

            sortdistindex = d.argsort(axis=0)[:self.k][::-1]
            pred_point = np.average(arrayY[sortdistindex])
            predY.append(pred_point)

        return predY

if __name__== "__main__":
    main()