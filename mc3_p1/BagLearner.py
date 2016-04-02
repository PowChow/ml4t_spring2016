"""
Bag learner based on template by Prof Tucker Balch
related to linear regression for machine learning for trading
gaTech
"""

import numpy as np
import math

class BagLearner(object):

    def __init__(self, learners = knn.Learner, kwargs ={"k":3}, bags =20, boost=False, verbose=False):
        self.verbose = verbose
        self.boost = boost

        self.learners = learners
        self.kwargs = kwargs
        self.bags = 20

        self.X = []
        self.Y = []
        self.X_bags = []

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner, append to existing
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.X.extend(dataX)
        self.Y.extend(dataY)

        tmpX = np.array(self.X)

        X_index = np.arange(start=0, stop=self.tmpX.shape[0])
        # if bagging selected then add distribution of points here
        for b in range(self.bags):
            #create bags here with data with replacement
            tmp_index = np.random.choice(X_index, size=X_index.shape[0], replace=True)
            self.X_bags.append(tmpX[tmp_index]) #adds new bag of values to list


    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        arrayX = np.array(self.X)
        arrayY = np.array(self.Y)

        #pseudocode
        for set in self.X_bags:
            #get preductions
            for learner in self.learners:
                predictions += learner.query(points)
            return average(predictions)


        #create zero array with size of points for Y predicted values
        predY = []
        t_dist = np.zeros(shape=(arrayX.shape[0],))

        # for loop to calculate Euclidean distance between query point and all
        for p in points:
            i = 0
            for x in arrayX:
                t_dist[i] = np.linalg.norm(p - x) #numpy calculate distance
                #d = pow((p - x), 2)
                #t_dist[i] = math.sqrt(np.sum(d))
                i= i+1

            sortdistindex = t_dist.argsort(axis=0)[:self.k]
            predY.append(np.average(t_dist[sortdistindex]))

            t_dist = np.zeros(shape=(arrayX.shape[0],))

        # if boosting add query here

        # return array of predicted Ys
        #print predY
        return predY

if __name__== "__main__":
