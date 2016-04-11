"""
Bag learner based on template by Prof Tucker Balch
related to linear regression for machine learning for trading
gaTech
"""

import numpy as np
import math
import LinRegLearner as lrl
import BagLearner as bl
import KNNLearner as knn

class BagLearner(object):

    def __init__(self, learner=knn.KNNLearner, kwargs={}, bags =20, boost=False, verbose=False):
        self.verbose = verbose
        self.boost = boost

        self.learner = learner
        self.kwargs = kwargs
        self.bags = 20

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner, append to existing
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.X = np.array(dataX)
        self.Y = np.array(dataY)
        self.X_bags = []
        self.Y_bags = []


        X_index = np.arange(start=0, stop=self.X.shape[0])
        # if bagging selected then add distribution of points here
        for b in range(self.bags):
            #create bags here with data with replacement
            tmp_index = np.random.choice(X_index, size=X_index.shape[0], replace=True)
            self.X_bags.append(self.X[tmp_index])
            self.Y_bags.append(self.Y[tmp_index])

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        arrayX = np.array(self.X)
        arrayY = np.array(self.Y)

        kwargs = self.kwargs
        learners_list = []
        pred = []
        for i in range(0,self.bags):
            #learners_list.append(self.learner(**kwargs))
            l = self.learner(**kwargs)
            l.addEvidence(self.X_bags[i], self.Y_bags[i])
            pred.append(l.query(points)) #outputs estimate for points related to a set of bags

        predY_bags = np.average(np.array(pred), axis=0)
        return predY_bags

if __name__== "__main__":
    main()