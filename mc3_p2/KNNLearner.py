"""
kNN learner based on template by Prof Tucker Balch
related to linear regression for machine learning for trading
gaTech
"""

import numpy as np
import math
import csv
from scipy.spatial import distance

class KNNLearner(object):

    def __init__(self, k=3, verbose=False):
        self.verbose = verbose
        self.k = k

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner, append to existing
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.X = np.array(dataX)
        self.Y = np.array(dataY)

        
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