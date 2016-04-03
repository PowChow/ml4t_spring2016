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

    def __init__(self, learner=knn.KNNLearner, kwargs ={"k":3}, bags =20, boost=False, verbose=False):
        self.verbose = verbose
        self.boost = boost

        self.learner = learner
        self.kwargs = kwargs
        self.bags = 20

        self.X_bags = []
        self.Y_bags = []

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner, append to existing
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.X = dataX
        self.Y = dataY

        tmpX = np.array(self.X)
        tmpY = np.array(self.Y)

        X_index = np.arange(start=0, stop=tmpX.shape[0])
        # if bagging selected then add distribution of points here
        for b in range(self.bags):
            #create bags here with data with replacement
            tmp_index = np.random.choice(X_index, size=X_index.shape[0], replace=True)
            self.X_bags.append(tmpX[tmp_index]) #adds new bag of values to list
            self.Y_bags.append(tmpY[tmp_index])

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
        for i in range(0,self.bags):
            learners_list.append(self.learner(**kwargs))

        pred = []
        for learn in learners_list:
            learn.addEvidence(self.X_bags[i], self.Y_bags[i])
            pred.append(learn.query(points)) #outputs estimate for points related to a set of bags

        pred_array = np.array(pred)

        predY_bags = np.average(pred_array, axis=0)
        return predY_bags

if __name__== "__main__":
    main()