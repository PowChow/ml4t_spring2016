"""
kNN learner based on template by Prof Tucker Balch
related to linear regression for machine learning for trading
gaTech
"""

import numpy as np
import math

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

        # if bagging selected then add distribution of points here

        #self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)

        
    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        arrayX = self.X
        arrayY = self.Y

        #create zero array with size of points for Y predicted values
        predY = []
        t_dist = []

        # for loop to calculate Euclidean distance between query point and all
        for p in points:
            for x in arrayX:
                t_dist.append(np.linalg.norm(p - x)) #numpy calculate distance

            t_dist_array = np.array(t_dist)
            sortdistindex = t_dist_array.argsort(axis=0)[:self.k][::-1]
            #print sortdistindex
            predY.append(np.average(arrayY[sortdistindex]))
            #predY.append(np.sum(arrayY[sortdistindex]) / self.k)

            t_dist = []


        # return array of predicted Y
        # print len(predY), predY
        return predY

if __name__== "__main__":
    main()